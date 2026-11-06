"""Hermes OpenAI-compatible tool-bridge API (tools are NEVER executed here)."""
from __future__ import annotations

import hmac
import os
import threading
from collections.abc import Callable
from typing import Any, Protocol

from fastapi import FastAPI, Header, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from .contract import InvalidRequest, ModelProtocolError, ToolRegistry, make_prompt
from .stream import StreamPump, collect_reply, completion_identity, completion_response, stream_frames
from .upstream import DeepSeekWebBackend


class Backend(Protocol):
    def iter_text(self, prompt: str): ...


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")  # Hermes sends optional provider kwargs.
    model: str = "deepseek-chat"
    messages: list[dict[str, Any]] = Field(min_length=1)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: Any = "auto"
    parallel_tool_calls: bool = True
    stream: bool = False
    search_enabled: bool = False
    thinking_enabled: bool = False
    temperature: float | None = None  # Web upstream does not support this.
    max_tokens: int | None = None
    reasoning_effort: str | None = None  # Accepted but not mapped to web API.


def _json_error(status: int, message: str, kind: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"type": kind, "message": message, "code": status}})


def _auth(authorization: str | None, key_override: str | None) -> None:
    expected = key_override if key_override is not None else os.environ.get("BRIDGE_API_KEY")
    if not expected:
        raise HTTPException(503, "Bridge API key is not configured")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing API bearer token")
    token = authorization[len("Bearer "):]
    # compare_digest(str, str) requires ASCII. Reject Unicode headers rather
    # than turning malformed authorization into an internal server error.
    if not token.isascii() or not expected.isascii() or not hmac.compare_digest(token, expected):
        raise HTTPException(401, "Invalid API bearer token")


def _num_workers() -> int:
    try:
        return min(32, max(1, int(os.environ.get("BRIDGE_MAX_INFLIGHT", "4"))))
    except ValueError:
        return 4


def create_app(
    backend_factory: Callable[[bool, bool], Backend] | None = None,
    *,
    api_key: str | None = None,
    model_name: str = "deepseek-chat",
    heartbeat_seconds: float = 8.0,
    max_inflight: int | None = None,
) -> FastAPI:
    if heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be positive")
    slots = threading.BoundedSemaphore(max_inflight or _num_workers())
    factory = backend_factory or (lambda think, search: DeepSeekWebBackend(thinking=think, search=search))
    app = FastAPI(title="DeepSeek4Web Hermes Tool Bridge", version="0.2.0")

    @app.exception_handler(HTTPException)
    async def http_error(_request, exc: HTTPException):
        kind = "authentication_error" if exc.status_code == 401 else "invalid_request_error"
        return _json_error(exc.status_code, str(exc.detail), kind)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request, _exc: RequestValidationError):
        return _json_error(400, "Invalid chat completion request", "invalid_request_error")

    @app.get("/health")
    def health():
        return {"status": "ok", "component": "hermes-tool-bridge"}

    @app.get("/v1/models")
    def models(authorization: str | None = Header(default=None)):
        _auth(authorization, api_key)
        return {"object": "list", "data": [{"id": model_name, "object": "model", "owned_by": "deepseek4web-bridge"}]}

    @app.post("/v1/chat/completions")
    def chat(payload: ChatRequest, authorization: str | None = Header(default=None)):
        _auth(authorization, api_key)
        if payload.model != model_name:
            raise HTTPException(400, "Model is not supported by this bridge")
        try:
            registry = ToolRegistry.from_request(payload.tools, payload.tool_choice, payload.parallel_tool_calls)
            prompt = make_prompt(payload.messages, registry)
        except (InvalidRequest, TypeError, ValueError, RecursionError, OverflowError) as exc:
            raise HTTPException(422, "Invalid tools, tool_choice, or message context") from exc

        if not slots.acquire(blocking=False):
            raise HTTPException(503, "Bridge is at concurrency capacity")
        identity = completion_identity(model_name)
        def start():
            return factory(payload.thinking_enabled, payload.search_enabled).iter_text(prompt)

        if payload.stream:
            try:
                pump = StreamPump(start, on_finish=slots.release, heartbeat=heartbeat_seconds).start()
            except Exception:
                slots.release()
                raise HTTPException(502, "Cannot start upstream worker") from None
            return StreamingResponse(
                stream_frames(pump, registry, identity),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
            )
        try:
            text, calls = collect_reply(start(), registry)
            return completion_response(identity, text, calls)
        except ModelProtocolError:
            return _json_error(502, "Model tool decision failed validation", "tool_protocol_error")
        except Exception:
            return _json_error(502, "Upstream chat could not be completed", "upstream_error")
        finally:
            slots.release()

    return app


app = create_app()
