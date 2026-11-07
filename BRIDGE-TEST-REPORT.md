# Hermes Tool Bridge test report

## Verification environment

- Date: 2026-11-07
- Python: 3.13 (local container)
- Tests: 60 passed (`python -m pytest -q tests_hermes_tool_bridge`)
- Compile check: `python -m compileall -q hermes_tool_bridge scripts tests_hermes_tool_bridge`
- Static GitHub read: `dsk/api.py` blob SHA `d73ff3c0bbc23eda9a91b972f6781adbbf26db79` on `main`.
- Hardening script anchors checked against that blob; transformation test uses a
  synthetic fixture. Actual repository checkout not available in the container.

## Covered by fake-upstream regression tests

- OpenAI-shaped non-stream and SSE answers; streamed argument deltas indexed and reassembled.
- Safe tool detection; ordinary prose and fenced tool JSON are not actionable.
- Per-request name allowlist, forced/none/required tool choice, parallel call caps.
- JSON Schema enforcement, no unknown parameters by default, refs denied.
- Missing, malformed, oversized, duplicate-key and invalid argument rejection.
- Bearer key authentication, error sanitization, invalid messages and model IDs.
- Two-turn mock Hermes executor roundtrip with role=tool and tool_call_id.
- Upstream exceptions, keepalives, cancellation signal, concurrency busy/recovery.
- Original dsk adapter mocked for chunk conversion, flags and generator cleanup.

## Not verified

- Live DeepSeek web API, actual authentication, Cloudflare cookie refresh, PoW.
- Runtime execution by the actual Hermes Agent (a contract-level stub was used).
- Deployment, distributed state, security penetration test, real load testing.
- Applied dsk/api.py hardening on a real local Git checkout.

The above are explicit release blockers for a production-grade deployment.
