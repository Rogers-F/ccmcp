# Claude Code MCP (ccmcp)

A FastMCP server that wraps the Claude Code CLI (`claude`) for seamless integration with MCP-compatible clients.

## Features

- **Non-interactive execution**: Run Claude Code tasks via `claude --print --output-format stream-json`
- **Session persistence**: Continue conversations using `SESSION_ID`
- **Sandbox modes**: Map security policies to Claude Code's `--tools` and `--permission-mode`
- **Image support**: Attach images via base64-encoded stream-json input
- **Windows compatible**: Handles `.cmd/.bat` wrappers and stdin prompt passing

## Requirements

- Python >= 3.12
- Claude Code CLI installed and available in PATH (`claude --version` >= 2.0)

## Installation

```bash
# Install from source
pip install git+https://github.com/Rogers-F/ccmcp.git

# Or install locally
git clone https://github.com/Rogers-F/ccmcp.git
cd ccmcp
pip install -e .
```

### Add to Claude Code

```bash
claude mcp add ccmcp -s user --transport stdio -- uvx --from git+https://github.com/Rogers-F/ccmcp.git ccmcp
```

## Tool Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `PROMPT` | str | required | Task instruction to send to Claude Code |
| `cd` | Path | `.` | Working directory for Claude Code |
| `sandbox` | Literal | `read-only` | Sandbox policy: `read-only`, `workspace-write`, `danger-full-access` |
| `SESSION_ID` | str | `""` | Resume session by UUID; empty starts new session |
| `return_all_messages` | bool | `False` | Return full stream-json event list |
| `image` | List[Path] | `None` | Attach images (base64 encoded) |
| `model` | str | `""` | Model override |
| `timeout_seconds` | int | `300` | Maximum execution time |

## Sandbox Mapping

| Sandbox Mode | Claude Code Flags |
|--------------|-------------------|
| `read-only` | `--tools Read,Glob,Grep,LS --permission-mode dontAsk` |
| `workspace-write` | `--tools Read,Edit,Glob,Grep,LS --permission-mode acceptEdits` |
| `danger-full-access` | `--tools default --permission-mode bypassPermissions` |

## Return Value

```json
{
  "success": true,
  "SESSION_ID": "uuid-string",
  "agent_messages": "Claude's response text",
  "all_messages": []  // Only when return_all_messages=True
}
```

Or on failure:

```json
{
  "success": false,
  "SESSION_ID": "uuid-string",
  "error": "Error description"
}
```

## License

MIT License - Copyright (c) 2025 guda.studio
