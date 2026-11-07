"""Hermes OpenAI-compatible hybrid tool gateway (never executes tools)."""
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
from .hybrid import routed_frames, wire_frames
from .native import NativeDeepSeekBackend, NativeUnsupported, NativeUnavailable
from .stream import StreamPump, collect_reply, completion_identity, completion_response
from .upstream import DeepSeekWebBackend


class Backend(Protocol):
    def iter_text(self, prompt: str): ...


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    model: str = "deepseek-chat"
    messages: list[dict[str, Any]] = Field(min_length=1)
    tools: list[dict[str, Any]] = Field(default_factory=list)
    tool_choice: Any = "auto"
    parallel_tool_calls: bool = True
    stream: bool = False
    search_enabled: bool = False
    thinking_enabled: bool = False
    temperature: float | None = None
    max_tokens: int | None = None
    reasoning_effort: str | None = None


def _json_error(status: int, message: str, kind: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"type": kind, "message": message, "code": status}})


def _auth(authorization: str | None, key_override: str | None) -> None:
    expected = key_override if key_override is not None else os.environ.get("BRIDGE_API_KEY")
    if not expected:
        raise HTTPException(503, "Bridge API key is not configured")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "Missing API bearer token")
    token = authorization[len("Bearer "):]
    if not token.isascii() or not expected.isascii() or not hmac.compare_digest(token, expected):
        raise HTTPException(401, "Invalid API bearer token")


def _num_workers() -> int:
    try:
        return min(32, max(1, int(os.environ.get("BRIDGE_MAX_INFLIGHT", "4"))))
    except ValueError:
        return 4


def _enabled(name: str) -> bool:
    return os.environ.get(name, "false").lower() in ("1", "true", "yes")


def create_app(
    backend_factory: Callable[[bool, bool], Backend] | None = None,
    *,
    native_factory: Callable[[], Any] | None = None,
    api_key: str | None = None,
    model_name: str = "deepseek-chat",
    heartbeat_seconds: float = 8.0,
    max_inflight: int | None = None,
    tool_mode: str | None = None,
    allow_web_fallback: bool | None = None,
) -> FastAPI:
    if heartbeat_seconds <= 0:
        raise ValueError("heartbeat_seconds must be positive")
    mode = (tool_mode or os.environ.get("BRIDGE_TOOL_MODE", "bridge")).strip().lower()
    if mode not in ("bridge", "native", "hybrid"):
        raise ValueError("BRIDGE_TOOL_MODE must be bridge, native, or hybrid")
    fallback = _enabled("BRIDGE_ALLOW_WEB_FALLBACK") if allow_web_fallback is None else allow_web_fallback
    slots = threading.BoundedSemaphore(max_inflight or _num_workers())
    web_factory = backend_factory or (lambda think, search: DeepSeekWebBackend(thinking=think, search=search))
    official_factory = native_factory or NativeDeepSeekBackend
    app = FastAPI(title="DeepSeek4Web Hermes Hybrid Gateway", version="0.3.0")

    @app.exception_handler(HTTPException)
    async def http_error(_request, exc: HTTPException):
        kind = "authentication_error" if exc.status_code == 401 else "invalid_request_error"
        return _json_error(exc.status_code, str(exc.detail), kind)

    @app.exception_handler(RequestValidationError)
    async def validation_error(_request, _exc: RequestValidationError):
        return _json_error(400, "Invalid chat completion request", "invalid_request_error")

    @app.get("/health")
    def health():
        return {"status": "ok", "component": "hermes-hybrid-gateway", "mode": mode}

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
            prompt = make_prompt(payload.messages, registry)  # common bounds and message validation
        except (InvalidRequest, TypeError, ValueError, RecursionError, OverflowError) as exc:
            raise HTTPException(422, "Invalid tools, tool_choice, or message context") from exc
        if mode == "native" and payload.search_enabled:
            raise HTTPException(422, "Native route does not provide DeepSeek web search")
        if mode == "hybrid" and payload.search_enabled and not fallback:
            raise HTTPException(422, "DeepSeek web search requires explicit web fallback")
        if not slots.acquire(blocking=False):
            raise HTTPException(503, "Bridge is at concurrency capacity")
        identity = completion_identity(model_name)

        def web():
            return web_factory(payload.thinking_enabled, payload.search_enabled)

        candidate = None
        if mode != "bridge" and not (mode == "hybrid" and payload.search_enabled):
            try:
                candidate = official_factory()
            except Exception:
                slots.release()
                return _json_error(503, "Native API is not configured", "configuration_error")
        if mode == "native" and (candidate is None or not getattr(candidate, "configured", True)):
            slots.release()
            return _json_error(503, "Native API key is not configured", "configuration_error")
        if mode == "hybrid" and not fallback and (candidate is None or not getattr(candidate, "configured", True)):
            slots.release()
            return _json_error(503, "Native API key is not configured", "configuration_error")

        if payload.stream:
            try:
                routed_mode = "bridge" if mode == "hybrid" and payload.search_enabled else mode
                producer = lambda: routed_frames(
                    routed_mode, candidate, web, registry=registry, identity=identity,
                    messages=payload.messages, tools=payload.tools, model=model_name,
                    prompt=prompt, fallback_allowed=fallback, temperature=payload.temperature,
                    max_tokens=payload.max_tokens,
                )
                pump = StreamPump(producer, on_finish=slots.release, heartbeat=heartbeat_seconds).start()
            except Exception:
                slots.release()
                return _json_error(502, "Cannot start upstream worker", "upstream_error")
            return StreamingResponse(
                wire_frames(pump), media_type="text/event-stream",
                headers={"Cache-Control": "no-cache, no-transform", "X-Accel-Buffering": "no"},
            )

        try:
            if mode == "bridge" or (mode == "hybrid" and payload.search_enabled):
                text, calls = collect_reply(web().iter_text(prompt), registry)
                return completion_response(identity, text, calls)
            if candidate is None or not getattr(candidate, "configured", True):
                if mode == "hybrid" and fallback:
                    text, calls = collect_reply(web().iter_text(prompt), registry)
                    return completion_response(identity, text, calls)
                raise NativeUnavailable("Native API key missing")
            try:
                result = candidate.complete(payload.messages, registry, payload.tools, model_name,
                                            payload.temperature, payload.max_tokens)
            except NativeUnsupported:
                if mode != "hybrid" or not fallback:
                    raise
                text, calls = collect_reply(web().iter_text(prompt), registry)
                return completion_response(identity, text, calls)
            return completion_response(identity, result.content, result.calls)
        except NativeUnavailable:
            return _json_error(503, "Native API key is not configured", "configuration_error")
        except NativeUnsupported:
            return _json_error(502, "Native function calling unsupported", "native_unsupported")
        except ModelProtocolError:
            return _json_error(502, "Model tool decision failed validation", "tool_protocol_error")
        except Exception:
            return _json_error(502, "Upstream chat could not be completed", "upstream_error")
        finally:
            slots.release()

    return app


app = create_app()
