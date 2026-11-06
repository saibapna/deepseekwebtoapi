"""Bounded SSE emitter with incremental plain-text and buffered tool decisions."""
from __future__ import annotations

import json
import queue
import threading
import time
import uuid
from collections.abc import Callable, Iterable, Iterator
from typing import Any

from .contract import (
    CLOSE_TAG,
    MAX_OUTPUT_BYTES,
    ModelProtocolError,
    OPEN_TAG,
    ToolRegistry,
)


def completion_identity(model: str) -> dict[str, Any]:
    return {"id": "chatcmpl-" + uuid.uuid4().hex, "created": int(time.time()), "model": model}


def completion_response(identity: dict[str, Any], text: str | None, calls: list[dict[str, Any]]) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": text}
    if calls:
        message["tool_calls"] = calls
    return {
        **identity,
        "object": "chat.completion",
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if calls else "stop"}],
    }


def _chunk(identity: dict[str, Any], delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
    return {
        **identity,
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def frame(value: dict[str, Any]) -> str:
    return "data: " + json.dumps(value, separators=(",", ":"), ensure_ascii=False) + "\n\n"


def error_frame(kind: str) -> str:
    # Do not disclose exception strings, raw responses, tool arguments or tokens.
    return frame({"error": {"type": kind, "message": "Bridge request could not be completed"}})


def _tool_delta_frames(identity: dict[str, Any], calls: list[dict[str, Any]]) -> Iterator[str]:
    for index, call in enumerate(calls):
        yield frame(_chunk(identity, {
            "tool_calls": [{
                "index": index, "id": call["id"], "type": "function",
                "function": {"name": call["function"]["name"], "arguments": ""},
            }]
        }))
        arguments = call["function"]["arguments"]
        for offset in range(0, len(arguments), 512):
            yield frame(_chunk(identity, {"tool_calls": [
                {"index": index, "function": {"arguments": arguments[offset:offset + 512]}}
            ]}))


class ReplyClassifier:
    """Only buffer when the initial output could be a tool envelope.

    Ordinary messages stream as soon as it is provable they do not start with
    OPEN_TAG. A proposed tool is withheld and validated atomically before any
    tool_call delta reaches Hermes.
    """
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.mode = "undecided" if registry.enabled else "text"
        self.buffer = ""
        self.size = 0

    def push(self, fragment: str) -> list[str]:
        if not isinstance(fragment, str):
            raise ModelProtocolError("Non-text upstream fragment")
        self.size += len(fragment.encode("utf-8"))
        if self.size > MAX_OUTPUT_BYTES:
            raise ModelProtocolError("Model response too large")
        if self.mode == "text":
            return [fragment] if fragment else []
        self.buffer += fragment
        if self.mode == "tool":
            return []
        probe = self.buffer.lstrip()
        if not probe or OPEN_TAG.startswith(probe):
            return []
        if probe.startswith(OPEN_TAG):
            self.mode = "tool"
            return []
        # Prefix differs from the control tag. Safe to stream as ordinary text.
        self.mode = "text"
        to_send = self.buffer
        self.buffer = ""
        return [to_send]

    def finish(self) -> tuple[list[str], list[dict[str, Any]]]:
        if self.mode == "tool":
            content, calls = self.registry.validate_response(self.buffer)
            if content is not None:
                raise ModelProtocolError("Tool tag produced ordinary content")
            return [], calls
        if self.mode == "undecided":
            # Incomplete reserved prefix is never exposed as normal text.
            probe = self.buffer.lstrip()
            if probe and OPEN_TAG.startswith(probe):
                raise ModelProtocolError("Incomplete reserved tool tag")
            if self.registry.choice not in ("auto", "none"):
                raise ModelProtocolError("Required tool call missing")
            return ([self.buffer] if self.buffer else []), []
        if self.registry.enabled and self.registry.choice not in ("auto", "none"):
            raise ModelProtocolError("Required tool call missing")
        return [], []


def collect_reply(chunks: Iterable[str], registry: ToolRegistry) -> tuple[str | None, list[dict[str, Any]]]:
    items: list[str] = []
    size = 0
    it = iter(chunks)
    try:
        for fragment in it:
            if not isinstance(fragment, str):
                raise ModelProtocolError("Non-text upstream fragment")
            size += len(fragment.encode("utf-8"))
            if size > MAX_OUTPUT_BYTES:
                raise ModelProtocolError("Model response too large")
            items.append(fragment)
    finally:
        close = getattr(it, "close", None)
        if callable(close):
            close()
    return registry.validate_response("".join(items))


class StreamPump:
    """Thread-to-iterator handoff, queue bounded for backpressure.

    A producer releases the concurrency slot only when its upstream work ends.
    On disconnect the consumer signals cancellation; network reads depend on
    the dsk request timeout to end promptly, hence the stability patch.
    """
    def __init__(self, start: Callable[[], Iterable[str]], on_finish: Callable[[], None], heartbeat: float = 8.0):
        self._queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=32)
        self._stop = threading.Event()
        self._start = start
        self._finish = on_finish
        self.heartbeat = heartbeat
        self._thread = threading.Thread(target=self._worker, name="hermes-bridge-upstream", daemon=True)

    def _put(self, item: tuple[str, Any]) -> None:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.2)
                return
            except queue.Full:
                continue

    def _worker(self) -> None:
        source = None
        try:
            source = iter(self._start())
            for fragment in source:
                if self._stop.is_set():
                    break
                self._put(("text", fragment))
        except BaseException:
            # Never send exception text to the client.
            self._put(("error", None))
        finally:
            if source is not None:
                try:
                    close = getattr(source, "close", None)
                    if callable(close):
                        close()
                except Exception:
                    pass
            self._put(("done", None))
            self._finish()

    def start(self) -> "StreamPump":
        self._thread.start()
        return self

    def events(self) -> Iterator[tuple[str, Any]]:
        try:
            while True:
                try:
                    event = self._queue.get(timeout=self.heartbeat)
                except queue.Empty:
                    yield ("heartbeat", None)
                    continue
                if event[0] == "done":
                    return
                yield event
        finally:
            self._stop.set()


def stream_frames(pump: StreamPump, registry: ToolRegistry, identity: dict[str, Any]) -> Iterator[str]:
    classifier = ReplyClassifier(registry)
    completed = False
    try:
        yield frame(_chunk(identity, {"role": "assistant"}))
        for kind, value in pump.events():
            if kind == "heartbeat":
                yield ": keepalive\n\n"
                continue
            if kind == "error":
                yield error_frame("upstream_error")
                return
            for text in classifier.push(value):
                if text:
                    yield frame(_chunk(identity, {"content": text}))
        tail, calls = classifier.finish()
        for text in tail:
            if text:
                yield frame(_chunk(identity, {"content": text}))
        if calls:
            yield from _tool_delta_frames(identity, calls)
        yield frame(_chunk(identity, {}, "tool_calls" if calls else "stop"))
        completed = True
    except ModelProtocolError:
        yield error_frame("tool_protocol_error")
    except Exception:
        yield error_frame("bridge_error")
    finally:
        # A consumer disconnect stops the producer on its next I/O boundary.
        pump._stop.set()
    if completed:
        yield "data: [DONE]\n\n"
    # No [DONE] after error: strict clients should surface the failure.
