# Hermes hybrid native/fallback v0.3 — local verification

Tested against the additive v0.2 Hermes Bridge implementation on the
`saibapna/deepseekwebtoapi` draft PR branch. No credentials were used.

## Commands

```sh
python -m compileall -q hermes_tool_bridge scripts tests_hermes_tool_bridge
python -m pytest -q tests_hermes_tool_bridge
```

**Results: 97 passed (Python 3.13.5, pytest 9.0.2), no failures.**

The baseline v0.2 suite (60 cases) remains passing. The 37 extra cases cover:
- Native JSON request and response parsing over httpx.MockTransport
- Native tool names, required/named tool choices, schema revalidation
- Duplicate JSON keys, NaN, invalid/unregistered arguments and malformed JSON
- Native SSE data deltas, fragmented tool arguments and missing `[DONE]`
- HTTP explicit unsupported-tools detection vs unauthorized, 429, 5xx and timeout
- Hybrid native-first routing, explicit consent for web fallback, unknown errors
- Streaming tool-call delta emission only after full validation
- Web-session result roundtrip via Hermes `role: tool` / `tool_call_id`
- Separate Web Search route and absent credentials policy

## Test scope

All upstream calls were **mocked** and FastAPI tests used an in-process
TestClient. No request was sent to an official DeepSeek API, DeepSeek Web,
Cloudflare, or a live Hermes Agent process. CI across Python 3.11–3.13 and
live tools/PoW/login must pass separately before production rollout.

## Release blockers

- [ ] Check GitHub Actions for the PR and inspect logs.
- [ ] Confirm native model ID and tool support with an official API account.
- [ ] Run actual Hermes Agent `tool_calls -> tool -> final answer` smoke test.
- [ ] Exercise DeepSeek web fallbacks with session expiry and upstream failures.
- [ ] Review secrets, network exposure, request body limits, and permissions.
- [ ] Opt in to `BRIDGE_ALLOW_WEB_FALLBACK` knowingly; do not automatically
      switch on auth errors, rate limits, timeouts, or malformed model replies.
