# Hermes Hybrid Gateway v0.3 — native + web fallback

## Scope

This is an **additive upgrade** to Hermes Tool Bridge v0.2 for
`saibapna/deepseekwebtoapi`. It does not modify `openai_api.py` or the
original `dsk/api.py`. Hermes connects through the existing OpenAI Chat
Completions-compatible endpoint on local port **8788**. The bridge **never
executes tools**; Hermes retains tool execution, permission prompts and logs.

## Why two routes?

```text
Hermes Agent (agent, approvals, tool execution)
    | OpenAI Chat Completions: messages, tools, tool_choice
    v
Bridge v0.3 (mandatory bearer auth, JSON Schema validation)
    |
    +-- native: DeepSeek official API, DEEPSEEK_API_KEY
    |      - actual OpenAI-compatible tools / tool_calls
    |      - tool arguments revalidated and assigned fresh Hermes IDs
    |
    +-- hybrid: try native first
           | explicit unsupported-tools or incompatible native tool name
           | AND BRIDGE_ALLOW_WEB_FALLBACK=true
           v
          web bridge: existing dsk.api.DeepSeekAPI + DEEPSEEK_AUTH_TOKEN
          - serializes conversation, including tool results, in bounded prompt
          - parses only full <hermes_tool_call> ... </hermes_tool_call> envelope
          - returns validated OpenAI tool_calls to Hermes
```

**Important:** Native refers to the *official DeepSeek Chat Completions API*,
not the undocumented DeepSeek web session. The web adapter does **not** have
verified native function calling.

## Environment and modes

| Variable | Default | Purpose |
| --- | --- | --- |
| `BRIDGE_TOOL_MODE` | `bridge` | `bridge`, `native`, or `hybrid` |
| `BRIDGE_ALLOW_WEB_FALLBACK` | `false` | Explicit cross-provider fallback consent in hybrid |
| `BRIDGE_API_KEY` | unset | Required shared key for Hermes -> bridge |
| `DEEPSEEK_API_KEY` | unset | Official API secret for native |
| `DEEPSEEK_NATIVE_MODEL` | `deepseek-v4-pro` | Official model ID; verify availability on your account |
| `DEEPSEEK_AUTH_TOKEN` | unset | Separate unofficial web-session token for fallback |
| `BRIDGE_MAX_INFLIGHT` | `4` | Bounded concurrency (1–32) |

The native client sends its API key **only** to
`https://api.deepseek.com/chat/completions`; redirects are disabled and proxy
environment variables are not trusted. The web client uses a separate web
session token. The API publicly advertises model `deepseek-chat` even when
`DEEPSEEK_NATIVE_MODEL` selects a different official upstream model.

For a **real hybrid** configuration, set both credentials and opt in:

```sh
export BRIDGE_TOOL_MODE=hybrid
export BRIDGE_ALLOW_WEB_FALLBACK=true
export BRIDGE_API_KEY='replace-with-a-long-random-secret'
export DEEPSEEK_API_KEY='your-official-api-key'
export DEEPSEEK_NATIVE_MODEL='deepseek-v4-pro'
export DEEPSEEK_AUTH_TOKEN='your-separate-web-session-token'

python -m pip install -r requirements.txt
python -m pip install -r requirements-hermes-tool-bridge.txt
python -m uvicorn hermes_tool_bridge.app:app --host 127.0.0.1 --port 8788
```

In Hermes, run `hermes model` and choose **Custom endpoint**:
`http://127.0.0.1:8788/v1`, model `deepseek-chat`, API key `BRIDGE_API_KEY`,
Chat Completions transport. Keep a 64K+ compatible model context configured
as required by your Hermes version.

### Fallback policy

- `bridge`: web-only behavior from v0.2. Never call native. Safe default.
- `native`: official API only; missing key is 503. No web fallback.
- `hybrid`: prefer official API. The web fallback is used **only if**
  `BRIDGE_ALLOW_WEB_FALLBACK=true` and (a) native is not configured,
  (b) the model returns an explicit, recognized `400/422 unsupported tools`
  response, or (c) Hermes requests a name that cannot be represented by the
  official API's function-name constraints.
- `hybrid` + `search_enabled=true` opts into the web route only if web fallback
  is explicitly enabled; official API search is not silently approximated.
- HTTP `401`, `403`, `429`, `500+`, timeouts, malformed JSON, bad arguments,
  missing tool results, truncation, unknown `400`, or mid-stream failures
  **never** trigger fallback. These fail closed, without a second model
  request, and without surfacing credentials or raw upstream bodies.
- In streaming mode no switch occurs after native data was emitted.
- Never rely on web fallback for guaranteed availability; the web service
  may change, challenge or reject a request at any time.

## Tool validation and streaming

Both routes apply the **same ToolRegistry**: caller-supplied allowlist,
`tool_choice` (`auto`, `none`, `required`, named), tool count and parallel
limits, max JSON size, JSON Schema Draft 2020-12 checks, no out-of-band refs,
no duplicate JSON keys, and no invalid numeric constants. Provider-proposed
calls are reissued with fresh opaque `call_...` IDs for Hermes; Hermes tool
results are included in subsequent conversation turns by `tool_call_id`.

Native sends structured `tools` and `messages` directly to the official API;
web fallback serializes this into an untrusted prompt. Neither invokes MCP,
terminal, shell, filesystem, or other executors.

**Native streaming:** requests real upstream SSE. Without tools, validated
content deltas stream incrementally to Hermes. When tools are available, the
bridge buffers the native reply until the full `tool_calls` decision is
validated, then emits standard `delta.tool_calls` fragments; if the model
instead replies in text, those buffered content deltas are released after
completion. This deliberate latency tradeoff prevents partial tool execution.

**Web streaming:** continues to use the v0.2 incremental text classifier and
buffered envelope. Both routes emit `: keepalive` SSE comments while idle,
`finish_reason: stop|tool_calls`, then `[DONE]` on success. Stream errors send
an `error` frame and omit `[DONE]`. A bounded queue provides backpressure,
and cancellation closes the iterator at the next network read boundary.

## Regression tests

```sh
python -m compileall -q hermes_tool_bridge scripts tests_hermes_tool_bridge
python -m pytest -q tests_hermes_tool_bridge
```

The suite uses **httpx.MockTransport**, a fake DeepSeek web generator, and an
in-process ASGI test client. It verifies malformed arguments, unknown tools,
force/required/none choices, multi-turn `role: tool`, native streamed argument
fragments, missing `[DONE]`, failure mapping, no retry on 401/429/5xx/timeout,
explicit web failover, and prevention of duplicate token/body leakage.
It does **not** perform live Hermes execution or DeepSeek web login/PoW tests.

## Release blockers and deploy notes

1. Review CI on Python 3.11/3.12/3.13 and confirm Hermes transport in your
   installed version; use a harmless read-only tool in a real agent session.
2. Verify that the official API model ID you configured supports tools, and
   that your API key has access. Native explicitly disables thinking on Chat Completions: forced/required tool
   selection needs non-thinking mode, and Hermes does not replay the official
   `reasoning_content` across tool-result turns. Do not assume native thinking
   is enabled by `thinking_enabled`.
3. Test web fallback with a legitimate user session, PoW and network expiry.
4. Apply the optional revision-locked `scripts/apply_dsk_stability.py` only
   after inspecting its diff, so web calls have timeouts and close on cancel.
5. Keep this server bound to `127.0.0.1`, or behind authenticated TLS with
   request-size limits. Do **not** publish the old `dsk/server.py` `/cookies`
   or `/html` endpoints.
6. Do not commit any session token, cookie, or API key. Keep `main` unchanged
   until staged live smoke tests and operational review succeed.

The proxy cannot promise that every DeepSeek web session remains operational;
no end-to-end live readiness is claimed from unit tests alone.
