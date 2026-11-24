@AGENTS.md

## Claude and Codex

Claude Code and Codex share the conventions in `AGENTS.md`. Claude coordinates the research: it plans experiments, reviews changes and interprets results. Codex implements and runs the engineering work in its own git worktree, through the `codex` MCP bridge from ARIS (`mcp-servers/codex-exec`). Review Codex's actual diff and command output, not its summary, before merging. Formal training still requires the user's explicit approval.
