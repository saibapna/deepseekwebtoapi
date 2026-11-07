# DeepSeek4Web -> Hermes Agent: Hermes Tool Bridge (v0.2 + hybrid v0.3)

**New:** `docs/HERMES-HYBRID-v0.3.md` documents native DeepSeek API tool calling,
opted-in web fallback, streaming, schema checks, and the v0.3 deployment mode.
The `bridge` configuration remains the backwards-compatible default.


This **additive patch** for `saibapna/deepseekwebtoapi` introduces a separate
OpenAI-compatible model endpoint. It does not replace the existing `openai_api.py`. Use port **8788** so the legacy API can continue using **8787**.

## Architecture

```text
Hermes Agent (tools / approvals / execution / memory)
    | POST /v1/chat/completions: messages, tools, tool_choice
    v
Hermes Tool Bridge :8788
    | authorize API key, validate request and function schemas
    | convert messages + tool definitions to bounded serialized prompt
    v
Existing dsk.api.DeepSeekAPI (web session, SSE, PoW)
    | response is ordinary text OR exact reserved tool envelope
    v
Hermes Tool Bridge
    | parse exact envelope -> validate name + arguments + choice
    | emit OpenAI tool_calls (non-stream or SSE delta.tool_calls)
    v
Hermes Agent decides permissions and executes tool
    | subsequent role=tool + tool_call_id are included in next request
    +-----------------------------------------------------------^
```

The bridge **does not execute** shell commands, read/write files, invoke MCP,
or forward any tool call to an executor. Hermes owns execution and approval.

## Installation (repository root)

Install a compatible Python version (3.10+ is recommended), ideally in a virtualenv.
Copy the `hermes_tool_bridge/`, `scripts/` and `tests_hermes_tool_bridge/` folders and the
`requirements-hermes-tool-bridge.txt` file from this package into the root
of `deepseekwebtoapi` beside `dsk/` and `openai_api.py`.

```sh
python -m pip install -r requirements.txt
python -m pip install -r requirements-hermes-tool-bridge.txt

# Run on a clean branch, optional but strongly recommended for bounded I/O:
python scripts/apply_dsk_stability.py . --diff  # checks; makes no changes
python scripts/apply_dsk_stability.py . --apply # modifies dsk/api.py
python -m compileall -q hermes_tool_bridge dsk

export BRIDGE_API_KEY='replace-with-a-long-random-secret'
export DEEPSEEK_AUTH_TOKEN='your-own-session-token'
export BRIDGE_MAX_INFLIGHT=4

python -m uvicorn hermes_tool_bridge.app:app --host 127.0.0.1 --port 8788
```

**The optional stability script is revision-locked** to `dsk/api.py` Git blob
`d73ff3c0bbc23eda9a91b972f6781adbbf26db79`. If your checkout differs,
it refuses to modify the file. Do not force blind text replacement. Inspect
`git diff --check` and `git diff -- dsk/api.py` after running it.

### Connect Hermes

The official Hermes Agent provider documentation supports OpenAI-compatible
custom endpoints. Run `hermes model`, select **Custom endpoint**, enter:

- API base URL: `http://127.0.0.1:8788/v1`
- Model: `deepseek-chat`
- API key: the exact value of `BRIDGE_API_KEY`
- Transport: **Chat Completions** (when asked)

Alternatively, merge the named-provider example from
`examples/hermes-config.yaml` into `~/.hermes/config.yaml`, put your key in
`~/.hermes/.env`, and select `custom:deepseek4web-bridge:deepseek-chat`.
The `hermes model` wizard is preferred to avoid version-specific config changes.

### Smoke test without executing a tool

```sh
curl -sS http://127.0.0.1:8788/v1/models \
  -H "Authorization: Bearer $BRIDGE_API_KEY"

curl -sS http://127.0.0.1:8788/v1/chat/completions \
  -H "Authorization: Bearer $BRIDGE_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-chat","messages":[{"role":"user","content":"Say hello"}],"stream":false}'
```

When `tools` are passed, DeepSeek web is asked to choose between ordinary text
and an exact reserved control envelope. Example *upstream model text* (not a
request for someone to execute anything):

```text
<hermes_tool_call>{"version":"1","tool_calls":[{"name":"read_file","arguments":{"path":"example.txt"}}]}</hermes_tool_call>
```

Only an exact, complete envelope at the start of the reply is recognized.
Markdown-fenced JSON or an envelope embedded in prose remains plain content.
Unknown names, duplicate JSON keys, wrong `tool_choice`, invalid JSON Schema
arguments, unsupported refs, oversized replies, and excess parallel calls
fail closed. When validated, the bridge returns an OpenAI-shaped
`finish_reason: "tool_calls"` response, which Hermes handles.

### Tool-result roundtrip

When Hermes sends the next chat request, it may include an assistant message
with `tool_calls` and one or more `role: "tool"` messages containing matching
`tool_call_id`. The bridge preserves those IDs and the tool outputs as serialized
conversation data before asking DeepSeek to produce the next assistant turn.
Each call creates a fresh web session; this does not emulate native DeepSeek
multi-turn sessions or native role boundaries.

## Streaming behavior

- `stream: false`: complete response and any requested tool calls returned once.
- `stream: true` plain answer: SSE `delta.content` is forwarded incrementally
  after enough prefix text arrives to establish it is **not** a tool envelope.
- `stream: true` tool decision: reserved envelope is buffered **until complete**
  and **validated**, then OpenAI `delta.tool_calls` fragments are emitted with
  `index`, generated `id`, `function.name` and incremental `function.arguments`.
  This prevents executing half a JSON object or leaking tool JSON as prose.
- A `: keepalive` SSE comment is emitted on idle streams (default 8 seconds).
- On success, `finish_reason` is `stop` or `tool_calls`, followed by `[DONE]`.
  Stream failures emit an OpenAI-shaped `error` SSE data frame, then end
  **without** `[DONE]` so clients do not mistake failure for success.
- On disconnect, the producer receives a cancellation signal. Unblocked
  cleanup depends on the upstream read timeout; use the optional stability patch.

The **web bridge mode** is text/tool-envelope-only: `dsk.api` does not accept
native tool definitions. **v0.3** adds official-API native tools as an independent
opt-in transport; read `docs/HERMES-HYBRID-v0.3.md` before enabling it.

## Validation and capacity limits

- Up to 96 registered tools per request.
- At most 4 tool calls in one response (or 1 if `parallel_tool_calls=false`).
- Strict JSON Schema checking via `jsonschema` (Draft 2020-12). `$ref`,
  `$dynamicRef`, `$recursiveRef`, `$id`, and `$dynamicAnchor` are rejected,
  including local refs, to prevent out-of-band schema loading and ambiguity.
- Undeclared properties are denied for object schemas with named properties
  unless the schema explicitly opts in with `additionalProperties`.
- Tool argument JSON capped at 16 KB per call; model output at 160 KB;
  total tool definitions at 180 KB; prompt at 260 KB; 128 messages maximum.
- Per-instance concurrency capped by `BRIDGE_MAX_INFLIGHT` (1 to 32, default 4).
  Busy calls fail with 503; the worker releases its slot after upstream cleanup.
- Bearer authorization is **mandatory** for `/v1/models` and
  `/v1/chat/completions`. An unset key fails closed with HTTP 503.
- Errors never echo raw upstream response text, argument values, or tokens.

## Compatibility and limitations

The bridge only supports text content, including OpenAI `[{type:"text",text:...}]`
parts. Images, audio, file objects, and the Responses API (`/v1/responses`)
are not provided. Optional OpenAI generation parameters such as temperature,
max_tokens, and reasoning_effort are accepted but **not enforced** by the
web backend. Thinking and web-search flags can be supplied via
`thinking_enabled` and `search_enabled` but have not been live verified.

The original repository relies on an unofficial web session and is subject to
upstream layout, Cloudflare, and PoW changes. Never publish its browser bypass
`/cookies` or `/html` endpoints; keep this API on loopback or behind an
authenticated trusted reverse proxy with TLS and request-body limits. Never
check browser cookies or session tokens into Git.

## Regression suite

```sh
python -m pytest -q tests_hermes_tool_bridge/
```

Tests inject a fake upstream backend. They cover tool allowlists, schema
validation, multiple calls, forced/disabled choices, malformed envelopes,
streaming delta assembly, heartbeats, cancellation signaling, auth, multi-turn
role=tool context, upstream error sanitization, and the optional stability-script
transformation. The original `dsk` transport is tested with a stub, **not**
against a live DeepSeek login. Live Hermes end-to-end behavior, actual upstream
response format, browser challenges, and load/penetration tests remain TODOs.

## Rollout checklist

1. Apply on a branch, run tests, review `git diff --check`.
2. Set bridge+DeepSeek secrets outside the repository; bind API to loopback.
3. Verify `/v1/models`, ordinary chat, and slow chat with SSE.
4. Run a harmless Hermes tool (`calculator` / read-only fixture), inspect
   the `tool_call_id`, and verify the next `role: tool` turn resumes.
5. Verify `tool_choice=none`, named/required tool choice, stream cancellation,
   401/429/upstream disconnect, and capacity control.
6. Only then enable additional tools behind Hermes approvals/guardrails.

See https://hermes-agent.nousresearch.com/docs/integrations/providers for the
current Hermes custom-provider configuration contract.
