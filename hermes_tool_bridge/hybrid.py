"""Pure frame routing: native first, opt-in web fallback on explicit unsupported-tools.

Neither route executes tools. Tool arguments are validated *before* any
``delta.tool_calls`` is sent to Hermes.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

from .contract import ModelProtocolError, ToolRegistry
from .native import NativeResult, NativeUnsupported
from .stream import (
    ReplyClassifier, StreamPump, _chunk, _tool_delta_frames, error_frame, frame,
)


def web_frames(fragments: Iterable[str], registry: ToolRegistry, identity: dict[str, Any]) -> Iterator[str]:
    classifier = ReplyClassifier(registry)
    source = iter(fragments)
    try:
        yield frame(_chunk(identity, {"role": "assistant"}))
        for fragment in source:
            for text in classifier.push(fragment):
                if text:
                    yield frame(_chunk(identity, {"content": text}))
        tail, calls = classifier.finish()
        for text in tail:
            if text:
                yield frame(_chunk(identity, {"content": text}))
        if calls:
            yield from _tool_delta_frames(identity, calls)
        yield frame(_chunk(identity, {}, "tool_calls" if calls else "stop"))
        yield "data: [DONE]\n\n"
    finally:
        close = getattr(source, "close", None)
        if callable(close):
            close()


def native_frames(events: Iterable[tuple[str, Any]], registry: ToolRegistry,
                  identity: dict[str, Any]) -> Iterator[str]:
    """Buffer native text while tools are enabled: tool deltas may arrive late.

    Without tools native content is forwarded incrementally with no duplicate
    final content; the provider's final frame is validated before [DONE].
    """
    source = iter(events)
    started = False
    finished = False
    try:
        for kind, data in source:
            if kind not in ("text", "finish"):
                raise ModelProtocolError("Unknown native stream event")
            if not started:
                started = True
                yield frame(_chunk(identity, {"role": "assistant"}))
            if kind == "text":
                if not isinstance(data, str):
                    raise ModelProtocolError("Invalid native text event")
                if not registry.enabled and data:
                    yield frame(_chunk(identity, {"content": data}))
            else:
                if finished or not isinstance(data, NativeResult):
                    raise ModelProtocolError("Invalid native stream completion")
                finished = True
                if registry.enabled and data.content:
                    for pos in range(0, len(data.content), 512):
                        yield frame(_chunk(identity, {"content": data.content[pos:pos + 512]}))
                if data.calls:
                    yield from _tool_delta_frames(identity, data.calls)
                yield frame(_chunk(identity, {}, "tool_calls" if data.calls else "stop"))
        if not finished:
            raise ModelProtocolError("Native stream ended early")
        yield "data: [DONE]\n\n"
    finally:
        close = getattr(source, "close", None)
        if callable(close):
            close()


def routed_frames(
    mode: str, native: Any, web: Any, *, registry: ToolRegistry,
    identity: dict[str, Any], messages: list[dict[str, Any]],
    tools: list[dict[str, Any]], model: str, prompt: str,
    fallback_allowed: bool, temperature: float | None = None,
    max_tokens: int | None = None,
) -> Iterator[str]:
    def fallback():
        return web_frames(web().iter_text(prompt), registry, identity)

    if mode == "bridge":
        try:
            yield from fallback()
        except ModelProtocolError:
            yield error_frame("tool_protocol_error")
        except Exception:
            yield error_frame("upstream_error")
        return

    if native is None or not getattr(native, "configured", True):
        if mode == "hybrid" and fallback_allowed:
            yield from routed_frames("bridge", None, web, registry=registry, identity=identity,
                                     messages=messages, tools=tools, model=model, prompt=prompt,
                                     fallback_allowed=False)
        else:
            yield error_frame("native_unavailable")
        return

    sent = False
    try:
        iterator = native_frames(native.stream_reply(messages, registry, tools, model, temperature, max_tokens),
                                 registry, identity)
        for value in iterator:
            sent = True
            yield value
    except NativeUnsupported:
        # Must not fail over once *any* native frame reached Hermes.
        if mode == "hybrid" and fallback_allowed and not sent:
            yield from routed_frames("bridge", None, web, registry=registry, identity=identity,
                                     messages=messages, tools=tools, model=model, prompt=prompt,
                                     fallback_allowed=False)
        else:
            yield error_frame("native_unsupported")
    except ModelProtocolError:
        yield error_frame("tool_protocol_error")
    except Exception:
        yield error_frame("upstream_error")


def wire_frames(pump: StreamPump) -> Iterator[str]:
    """Emit a bounded producer's already-framed SSE values with heartbeats."""
    for kind, value in pump.events():
        if kind == "heartbeat":
            yield ": keepalive\n\n"
        elif kind == "error":
            yield error_frame("upstream_error")
        elif kind == "text":
            yield value
