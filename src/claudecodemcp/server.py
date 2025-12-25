"""FastMCP server implementation for wrapping the Claude Code CLI (`claude`)."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import queue
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path
from typing import Annotated, Any, Dict, Generator, List, Literal, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP("Claude Code MCP Server-from guda.studio")

ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp"}
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB per image
DEFAULT_TIMEOUT_SECONDS = 300  # 5 minutes


def _sandbox_to_claude_flags(
    sandbox: Literal["read-only", "workspace-write", "danger-full-access"],
    *,
    dangerously_skip_permissions: bool,
) -> list[str]:
    """Map a Codex-style sandbox mode to Claude Code CLI flags.

    Verified via `claude --help`:
    - `--permission-mode` choices are camelCase:
      `acceptEdits`, `bypassPermissions`, `default`, `delegate`, `dontAsk`, `plan`
    - `--tools` accepts `default` or a comma-separated list, e.g. `"Bash,Edit,Read"`.
    """
    if sandbox == "read-only":
        flags = [
            "--tools",
            "Read,Glob,Grep,LS",
            "--permission-mode",
            "dontAsk",
        ]
    elif sandbox == "workspace-write":
        flags = [
            "--tools",
            "Read,Edit,Glob,Grep,LS",
            "--permission-mode",
            "acceptEdits",
        ]
    else:
        flags = [
            "--tools",
            "default",
            "--permission-mode",
            "bypassPermissions",
        ]

    if dangerously_skip_permissions:
        flags.extend(
            [
                "--allow-dangerously-skip-permissions",
                "--dangerously-skip-permissions",
            ]
        )

    return flags


def _guess_media_type(image_path: Path) -> str:
    """Best-effort image MIME type inference for stream-json image blocks."""
    suffix = image_path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".gif":
        return "image/gif"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".bmp":
        return "image/bmp"
    return "image/png"


def _validate_image(image_path: Path, workspace: Path) -> Optional[str]:
    """Validate image file. Returns error message or None if valid."""
    if not image_path.exists() or not image_path.is_file():
        return f"Image file does not exist: {image_path}"

    suffix = image_path.suffix.lower()
    if suffix not in ALLOWED_IMAGE_EXTENSIONS:
        return f"Invalid image extension '{suffix}'. Allowed: {ALLOWED_IMAGE_EXTENSIONS}"

    try:
        size = image_path.stat().st_size
        if size > MAX_IMAGE_SIZE_BYTES:
            return f"Image too large: {size} bytes (max {MAX_IMAGE_SIZE_BYTES})"
    except OSError as e:
        return f"Cannot read image file: {e}"

    try:
        resolved = image_path.resolve()
        workspace_resolved = workspace.resolve()
        if not str(resolved).startswith(str(workspace_resolved)):
            return f"Image must be within workspace directory: {workspace}"
    except Exception:
        pass

    return None


def _build_stream_json_stdin_payload(prompt: str, images: List[Path]) -> str:
    """Build stdin payload for `--input-format stream-json` (NDJSON)."""
    blocks: list[dict[str, Any]] = []
    for image_path in images:
        data_b64 = base64.b64encode(image_path.read_bytes()).decode("ascii")
        blocks.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": _guess_media_type(image_path),
                    "data": data_b64,
                },
            }
        )

    blocks.append({"type": "text", "text": prompt})

    message = {
        "type": "user",
        "session_id": "",
        "message": {"role": "user", "content": blocks},
        "parent_tool_use_id": None,
    }

    return json.dumps(message, ensure_ascii=False) + "\n"


def _extract_assistant_text(event: Dict[str, Any]) -> str:
    """Extract assistant text content from a Claude Code `stream-json` event."""
    if event.get("type") != "assistant":
        return ""

    message = event.get("message")
    if not isinstance(message, dict):
        return ""

    content = message.get("content")
    if isinstance(content, str):
        return content

    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") != "text":
            continue
        text = block.get("text")
        if isinstance(text, str) and text:
            parts.append(text)
    return "".join(parts)


def _find_claude_executable() -> Optional[str]:
    """Find the Claude CLI executable, handling Windows .cmd/.bat wrappers."""
    claude_path = shutil.which("claude")
    if claude_path:
        return claude_path
    if os.name == "nt":
        for ext in [".cmd", ".bat", ".exe"]:
            path = shutil.which(f"claude{ext}")
            if path:
                return path
    return None


def run_shell_command(
    cmd: list[str],
    *,
    cwd: Path,
    stdin_text: str,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
) -> Generator[str, None, None]:
    """Execute a command and stream merged stdout/stderr line-by-line.

    Completion event (Codex `turn.completed` equivalent):
    - In `--output-format stream-json` mode, Claude Code emits a final event
      with `type == "result"`. This signals execution completion.
    """
    popen_cmd = cmd.copy()
    claude_path = _find_claude_executable()
    if not claude_path:
        yield json.dumps({"type": "error", "message": "Claude CLI not found in PATH"})
        return

    popen_cmd[0] = claude_path

    # Windows .cmd/.bat requires shell=True or cmd.exe wrapper
    use_shell = False
    if os.name == "nt" and claude_path.lower().endswith((".cmd", ".bat")):
        use_shell = True

    try:
        process = subprocess.Popen(
            popen_cmd,
            shell=use_shell,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            encoding="utf-8",
            errors="replace",  # Handle non-UTF8 gracefully
            cwd=str(cwd),
        )
    except FileNotFoundError as e:
        yield json.dumps({"type": "error", "message": f"Failed to start Claude CLI: {e}"})
        return
    except OSError as e:
        yield json.dumps({"type": "error", "message": f"OS error starting Claude CLI: {e}"})
        return

    if process.stdin:
        try:
            text = stdin_text
            if not text.endswith("\n"):
                text += "\n"
            process.stdin.write(text)
            process.stdin.flush()
        except BrokenPipeError:
            pass
        finally:
            try:
                process.stdin.close()
            except Exception:
                pass

    output_queue: queue.Queue[str | None] = queue.Queue()
    GRACEFUL_SHUTDOWN_DELAY = 0.3
    start_time = time.monotonic()

    def is_execution_completed(line: str) -> bool:
        try:
            data = json.loads(line)
            return data.get("type") == "result"
        except (json.JSONDecodeError, AttributeError, TypeError):
            return False

    def read_output() -> None:
        try:
            if process.stdout:
                for line in iter(process.stdout.readline, ""):
                    stripped = line.strip()
                    output_queue.put(stripped)
                    if is_execution_completed(stripped):
                        time.sleep(GRACEFUL_SHUTDOWN_DELAY)
                        if process.poll() is None:
                            process.terminate()
                        break
                process.stdout.close()
        except Exception:
            pass
        finally:
            output_queue.put(None)

    thread = threading.Thread(target=read_output, daemon=True)
    thread.start()

    while True:
        elapsed = time.monotonic() - start_time
        if elapsed > timeout_seconds:
            process.kill()
            yield json.dumps({
                "type": "error",
                "message": f"Execution timed out after {timeout_seconds} seconds",
            })
            break

        try:
            line = output_queue.get(timeout=0.5)
            if line is None:
                break
            yield line
        except queue.Empty:
            if process.poll() is not None and not thread.is_alive():
                break

    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
    thread.join(timeout=5)

    while not output_queue.empty():
        try:
            line = output_queue.get_nowait()
            if line is not None:
                yield line
        except queue.Empty:
            break


def _run_claude_session(
    *,
    PROMPT: str,
    cd: Path,
    sandbox: Literal["read-only", "workspace-write", "danger-full-access"],
    SESSION_ID: str,
    return_all_messages: bool,
    image: Optional[List[Path]],
    model: str,
    dangerously_skip_permissions: bool,
    no_session_persistence: bool,
    include_partial_messages: bool,
    timeout_seconds: int,
) -> Dict[str, Any]:
    """Synchronous implementation of the Claude Code session runner."""
    images = image or []

    if not PROMPT or not PROMPT.strip():
        return {"success": False, "error": "PROMPT must not be empty."}

    if not cd.exists() or not cd.is_dir():
        return {
            "success": False,
            "error": f"Working directory does not exist or is not a directory: {cd}",
        }

    claude_path = _find_claude_executable()
    if not claude_path:
        return {
            "success": False,
            "error": "Claude Code CLI not found. Please install it and ensure 'claude' is in PATH.",
        }

    for image_path in images:
        err = _validate_image(image_path, cd)
        if err:
            return {"success": False, "error": err}

    requested_session_id = SESSION_ID.strip()
    resume = bool(requested_session_id)
    session_id = requested_session_id or str(uuid.uuid4())

    try:
        if images:
            input_format = "stream-json"
            stdin_payload = _build_stream_json_stdin_payload(PROMPT, images)
        else:
            input_format = "text"
            stdin_payload = PROMPT
    except Exception as exc:
        return {"success": False, "error": f"Failed to prepare stdin payload: {exc}"}

    cmd = [
        "claude",
        "--print",
        "--output-format",
        "stream-json",
        "--verbose",
        "--input-format",
        input_format,
    ]

    if include_partial_messages:
        cmd.append("--include-partial-messages")

    if no_session_persistence:
        cmd.append("--no-session-persistence")

    cmd.extend(
        _sandbox_to_claude_flags(
            sandbox,
            dangerously_skip_permissions=dangerously_skip_permissions,
        )
    )

    if model:
        cmd.extend(["--model", model])

    if resume:
        cmd.extend(["--resume", session_id])
    else:
        cmd.extend(["--session-id", session_id])

    all_messages: list[Dict[str, Any]] = []
    agent_messages = ""
    final_result_text: Optional[str] = None
    saw_result_event = False
    success = True
    err_message = ""

    for line in run_shell_command(cmd, cwd=cd, stdin_text=stdin_payload, timeout_seconds=timeout_seconds):
        if not line:
            continue

        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            err_message += "\n\n[non-json output] " + line
            continue

        if not isinstance(event, dict):
            continue

        all_messages.append(event)

        if event.get("type") == "error":
            success = False
            err_message += f"\n\n[cli error] {event.get('message', 'Unknown error')}"
            continue

        event_session_id = event.get("session_id")
        if isinstance(event_session_id, str) and event_session_id.strip():
            session_id = event_session_id

        event_type = event.get("type")

        if event_type == "assistant":
            agent_messages += _extract_assistant_text(event)

        if event_type == "result":
            saw_result_event = True

            result_text = event.get("result")
            if isinstance(result_text, str) and result_text:
                final_result_text = result_text

            subtype = event.get("subtype")
            is_error = bool(event.get("is_error"))
            if is_error or (isinstance(subtype, str) and subtype != "success"):
                success = False
                if isinstance(subtype, str) and subtype:
                    err_message += f"\n\n[claude result] subtype={subtype}"

    if final_result_text:
        agent_messages = final_result_text

    if not saw_result_event:
        success = False
        err_message = (
            "Did not receive the final Claude Code `result` event; "
            "unable to confirm completion.\n" + err_message
        )

    if not agent_messages:
        success = False
        err_message = (
            "Failed to extract agent output. "
            "Set return_all_messages=True to inspect the full event stream.\n"
            + err_message
        )

    if success:
        result: Dict[str, Any] = {
            "success": True,
            "SESSION_ID": session_id,
            "agent_messages": agent_messages,
        }
    else:
        result = {
            "success": False,
            "SESSION_ID": session_id,
            "error": err_message,
        }

    if return_all_messages:
        result["all_messages"] = all_messages

    return result


@mcp.tool(
    name="claude",
    description="""
Executes a non-interactive Claude Code session via the `claude` CLI.

Core mode:
- Uses `claude --print --output-format stream-json --verbose`
- Detects completion via the `type == "result"` event (Codex `turn.completed` equivalent)
- Supports session continuation via `SESSION_ID` and `--resume`

Prompt input:
- Uses stdin by default (avoids Windows CLI escaping/length limits)
- If `image` is provided, switches to `--input-format stream-json` and sends images as base64 blocks.
""",
)
async def claude(
    PROMPT: Annotated[str, "Task instruction to send to Claude Code."],
    cd: Annotated[
        Path,
        Field(
            description="Working directory for Claude Code (subprocess cwd). Must exist."
        ),
    ] = Path("."),
    sandbox: Annotated[
        Literal["read-only", "workspace-write", "danger-full-access"],
        Field(description="Sandbox policy mapped to --tools/--permission-mode."),
    ] = "read-only",
    SESSION_ID: Annotated[
        str,
        "Resume the specified session ID (UUID). Empty means start a new session.",
    ] = "",
    return_all_messages: Annotated[
        bool,
        "Return the full stream-json event list (for debugging/reasoning/tool events).",
    ] = False,
    image: Annotated[
        Optional[List[Path]],
        Field(
            description="Attach images via stream-json input with base64-encoded blocks.",
        ),
    ] = None,
    model: Annotated[
        str,
        Field(
            description="Model override (only use when explicitly requested by the user)."
        ),
    ] = "",
    dangerously_skip_permissions: Annotated[
        bool,
        Field(
            description="Enable --dangerously-skip-permissions (high risk; only for isolated sandboxes).",
        ),
    ] = False,
    no_session_persistence: Annotated[
        bool,
        Field(description="Disable session persistence on disk."),
    ] = False,
    include_partial_messages: Annotated[
        bool,
        Field(
            description="Include partial message chunks (--include-partial-messages).",
        ),
    ] = False,
    timeout_seconds: Annotated[
        int,
        Field(description="Maximum execution time in seconds. Default 300 (5 min)."),
    ] = DEFAULT_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    return await asyncio.to_thread(
        _run_claude_session,
        PROMPT=PROMPT,
        cd=cd,
        sandbox=sandbox,
        SESSION_ID=SESSION_ID,
        return_all_messages=return_all_messages,
        image=image,
        model=model,
        dangerously_skip_permissions=dangerously_skip_permissions,
        no_session_persistence=no_session_persistence,
        include_partial_messages=include_partial_messages,
        timeout_seconds=timeout_seconds,
    )


def run() -> None:
    """Start the MCP server over stdio transport."""
    mcp.run(transport="stdio")
