# Hermes Tool Bridge release notes

## v0.2.0

- Separate OpenAI-compatible Chat Completions gateway; does not modify legacy API.
- Strict tagged tool-decision protocol; JSON Schema validation, allowlist,
  `tool_choice`, capped parallel calls, per-request IDs.
- Real incremental SSE for ordinary text, atomic validated SSE tool-call deltas,
  keepalive comments, bounded thread/queue, cancellation signal.
- Preserve assistant tool_calls and role=tool messages across Hermes turns.
- Optional revision-locked dsk stability patch (timeouts, stream close, sanitize).
- Offline regression test suite with injected fake backend and mocked dsk.

Not included: remote deployment, bypass service security fixes, native upstream
function API, live Hermes/DeepSeek integration verification, and tool execution.
