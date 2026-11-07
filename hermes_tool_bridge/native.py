"""Opt-in native DeepSeek Chat Completions transport.

Only the official DeepSeek API host receives DEEPSEEK_API_KEY. Responses are
untrusted; the same Hermes tool registry validates native and fallback calls.
"""
from __future__ import annotations

import json
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import httpx

from .contract import (
    MAX_ARGUMENT_BYTES, MAX_CALLS, MAX_OUTPUT_BYTES, ModelProtocolError,
    ToolRegistry, _invalid_constant, _no_duplicate_keys, normalize_messages,
)

API_URL = "https://api.deepseek.com/chat/completions"
NATIVE_TOOL_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,128}$")
MAX_NATIVE_RESPONSE = 480_000
MAX_NATIVE_STREAM = 4_000_000


class NativeUnavailable(RuntimeError):
    """The native credential has not been configured."""


class NativeUnsupported(RuntimeError):
    """Native tool calling is explicitly unsupported (eligible for opted-in fallback)."""


class NativeFailure(RuntimeError):
    """Any other native request failure; never automatically retry via web."""


@dataclass(frozen=True)
class NativeResult:
    content: str | None
    calls: list[dict[str, Any]]


def _decode_json(raw: str) -> Any:
    try:
        return json.loads(raw, object_pairs_hook=_no_duplicate_keys, parse_constant=_invalid_constant)
    except (ValueError, TypeError, UnicodeError, OverflowError, RecursionError) as exc:
        raise ModelProtocolError("Invalid JSON from native API") from exc


def _unsupported_response(response: httpx.Response) -> bool:
    """Fail over only on explicit unsupported-tools 400/422 responses.

    Never fail over on auth, rate-limit, HTTP 5xx, timeout, or arbitrary 4xx.
    An unknown 400 may be a real request error and must stay visible as a 502.
    """
    if response.status_code not in (400, 422):
        return False
    # Bounded read and narrow matching; no response body is returned to clients.
    try:
        chunks = []
        used = 0
        for part in response.iter_bytes():
            used += len(part)
            if used > 8192:
                return False
            chunks.append(part)
        data = _decode_json(b"".join(chunks).decode("utf-8"))
    except (ModelProtocolError, UnicodeError):
        return False
    detail = data.get("error", {}) if isinstance(data, dict) else {}
    if not isinstance(detail, dict):
        return False
    message = detail.get("message")
    if not isinstance(message, str):
        return False
    message = message.lower()
    return any(phrase in message for phrase in (
        "tools not supported", "tool calling not supported", "tool calls not supported",
        "function calling not supported", "model does not support tools",
        "unknown parameter: tools", "unsupported parameter: tools",
        "tool_choice is not supported", "tool_choice not supported",
    ))


def _check_http_status(response: httpx.Response) -> None:
    if response.status_code == 200:
        return
    if _unsupported_response(response):
        raise NativeUnsupported("Native API explicitly does not support tools")
    raise NativeFailure("Native API request failed")


def _prepare_payload(
    messages: list[dict[str, Any]], registry: ToolRegistry, tools: list[dict[str, Any]],
    model: str, temperature: float | None, max_tokens: int | None, stream: bool,
) -> dict[str, Any]:
    normalized = normalize_messages(messages)
    for message in normalized:
        if message["role"] == "developer":
            message["role"] = "system"  # DeepSeek Chat Completions uses system/user/assistant/tool
        if message["role"] == "assistant" and "tool_calls" in message and not message["content"]:
            message["content"] = None
    # Explicitly disable DeepSeek thinking for reliable Hermes Chat Completions
    # tool roundtrips: forced tool_choice requires non-thinking mode, and Hermes
    # does not replay provider-specific reasoning_content in tool history.
    payload: dict[str, Any] = {"model": model, "messages": normalized, "stream": stream,
                               "thinking": {"type": "disabled"}}
    if temperature is not None:
        if not 0 <= temperature <= 2:
            raise ModelProtocolError("Unsupported native temperature")
        payload["temperature"] = temperature
    if max_tokens is not None:
        if not 1 <= max_tokens <= 65536:
            raise ModelProtocolError("Unsupported native max_tokens")
        payload["max_tokens"] = max_tokens
    if registry.enabled:
        if any(not NATIVE_TOOL_NAME.fullmatch(name) for name in registry.schemas):
            raise NativeUnsupported("Native model cannot represent these tool names")
        descriptions = {
            tool["function"]["name"]: tool["function"].get("description", "") for tool in tools
        }
        payload["tools"] = [{"type": "function", "function": {
            "name": name, "description": descriptions[name], "parameters": schema,
        }} for name, schema in registry.schemas.items()]
        payload["tool_choice"] = (
            {"type": "function", "function": {"name": registry.forced_name}}
            if registry.choice == "named" else registry.choice
        )
        payload["parallel_tool_calls"] = registry.parallel
    return payload


class _Accumulator:
    """Reassembles native content and fragmented tool deltas without trusting them."""
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.text: list[str] = []
        self.total_text_bytes = 0
        self.tools: dict[int, dict[str, str]] = {}
        self.finish_reason: str | None = None

    def add(self, event: Any) -> list[str]:
        if not isinstance(event, dict) or not isinstance(event.get("choices"), list):
            raise ModelProtocolError("Malformed native SSE event")
        choices = event["choices"]
        if not choices:  # usage-only trailer
            return []
        if len(choices) != 1 or not isinstance(choices[0], dict) or choices[0].get("index", 0) != 0:
            raise ModelProtocolError("Unexpected native stream choice")
        choice = choices[0]
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            raise ModelProtocolError("Malformed native delta")
        if self.finish_reason is not None and (delta.get("content") or delta.get("tool_calls")):
            raise ModelProtocolError("Native stream continued after finish")
        fragment = delta.get("content")
        output = []
        if fragment is not None:
            if not isinstance(fragment, str):
                raise ModelProtocolError("Invalid native content")
            self.total_text_bytes += len(fragment.encode("utf-8"))
            if self.total_text_bytes > MAX_OUTPUT_BYTES:
                raise ModelProtocolError("Native content exceeds output limit")
            self.text.append(fragment)
            if fragment:
                output.append(fragment)
        fragments = delta.get("tool_calls", [])
        if fragments:
            if not isinstance(fragments, list):
                raise ModelProtocolError("Malformed tool delta")
            for part in fragments:
                if not isinstance(part, dict):
                    raise ModelProtocolError("Malformed native tool fragment")
                index = part.get("index")
                if type(index) is not int or index < 0 or index >= (MAX_CALLS if self.registry.parallel else 1):
                    raise ModelProtocolError("Invalid tool index")
                entry = self.tools.setdefault(index, {"id": "", "name": "", "arguments": ""})
                if "id" in part:
                    value = part["id"]
                    if not isinstance(value, str) or not value or (entry["id"] and entry["id"] != value):
                        raise ModelProtocolError("Invalid native tool ID")
                    entry["id"] = value
                if "type" in part and part["type"] != "function":
                    raise ModelProtocolError("Unsupported native tool type")
                fn = part.get("function", {})
                if not isinstance(fn, dict):
                    raise ModelProtocolError("Malformed native function delta")
                for key in ("name", "arguments"):
                    if key in fn:
                        v = fn[key]
                        if not isinstance(v, str):
                            raise ModelProtocolError("Malformed native function field")
                        entry[key] += v
                if len(entry["arguments"].encode("utf-8")) > MAX_ARGUMENT_BYTES:
                    raise ModelProtocolError("Native arguments exceed limit")
                if len(entry["name"]) > 128 or len(entry["id"]) > 128:
                    raise ModelProtocolError("Native tool metadata exceeds limit")
        finish = choice.get("finish_reason")
        if finish is not None:
            if self.finish_reason is not None or finish not in ("stop", "tool_calls"):
                raise ModelProtocolError("Incomplete or conflicting native finish")
            self.finish_reason = finish
        return output

    def finish(self) -> NativeResult:
        if self.finish_reason is None:
            raise ModelProtocolError("Missing native finish_reason")
        if self.tools and set(self.tools) != set(range(len(self.tools))):
            raise ModelProtocolError("Non-contiguous native tool indexes")
        calls = []
        for index in sorted(self.tools):
            item = self.tools[index]
            if not item["id"] or not item["name"]:
                raise ModelProtocolError("Incomplete native tool metadata")
            calls.append({"id": item["id"], "type": "function", "function": {
                "name": item["name"], "arguments": item["arguments"]
            }})
        return _validate_reply({"role": "assistant", "content": "".join(self.text), "tool_calls": calls},
                               self.finish_reason, self.registry)


def _validate_reply(message: Any, finish: Any, registry: ToolRegistry) -> NativeResult:
    if not isinstance(message, dict) or message.get("role") != "assistant":
        raise ModelProtocolError("Invalid native message role")
    raw_content = message.get("content")
    if raw_content is not None and not isinstance(raw_content, str):
        raise ModelProtocolError("Invalid native assistant content")
    if raw_content is not None and len(raw_content.encode("utf-8")) > MAX_OUTPUT_BYTES:
        raise ModelProtocolError("Native content too large")
    raw_calls = message.get("tool_calls") or []
    if not isinstance(raw_calls, list):
        raise ModelProtocolError("Native tool_calls must be an array")
    if finish not in ("stop", "tool_calls") or bool(raw_calls) != (finish == "tool_calls"):
        raise ModelProtocolError("Native finish_reason does not match tool_calls")
    if raw_calls:
        # Shared schema/allowlist/choice checks; fresh Hermes IDs prevent exposing
        # untrusted native tool IDs to the agent's execution layer.
        calls = registry.format_native_calls(raw_calls)
        return NativeResult(content=None, calls=calls)
    if registry.enabled and registry.choice in ("required", "named"):
        raise ModelProtocolError("Native model omitted required tool call")
    return NativeResult(content=raw_content or "", calls=[])


class NativeDeepSeekBackend:
    """Dedicated official API transport. Never send session tokens to this host."""
    def __init__(self, *, api_key: str | None = None, transport: httpx.BaseTransport | None = None,
                 timeout: float = 75.0, model: str | None = None):
        self.api_key = api_key if api_key is not None else os.getenv("DEEPSEEK_API_KEY")
        self.transport = transport
        self.timeout = timeout
        self.model = model if model is not None else os.getenv("DEEPSEEK_NATIVE_MODEL", "deepseek-v4-pro")

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def _client(self) -> httpx.Client:
        if not self.api_key:
            raise NativeUnavailable("DEEPSEEK_API_KEY not configured")
        return httpx.Client(
            timeout=httpx.Timeout(self.timeout, connect=8.0),
            transport=self.transport, follow_redirects=False, trust_env=False,
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
        )

    def complete(self, messages, registry, tools, model, temperature=None, max_tokens=None) -> NativeResult:
        payload = _prepare_payload(messages, registry, tools, self.model, temperature, max_tokens, stream=False)
        try:
            with self._client() as client:
                with client.stream("POST", API_URL, json=payload) as response:
                    _check_http_status(response)
                    chunks = []
                    size = 0
                    for part in response.iter_bytes():
                        size += len(part)
                        if size > MAX_NATIVE_RESPONSE:
                            raise ModelProtocolError("Native response exceeds limit")
                        chunks.append(part)
                    raw = b"".join(chunks).decode("utf-8")
        except httpx.RequestError as exc:
            raise NativeFailure("Native API transport error") from exc
        data = _decode_json(raw)
        if not isinstance(data, dict) or not isinstance(data.get("choices"), list) or len(data["choices"]) != 1:
            raise ModelProtocolError("Invalid native choices")
        first = data["choices"][0]
        if not isinstance(first, dict) or first.get("index", 0) != 0:
            raise ModelProtocolError("Invalid native choice")
        return _validate_reply(first.get("message"), first.get("finish_reason"), registry)

    def stream_reply(self, messages, registry, tools, model, temperature=None, max_tokens=None) -> Iterator[tuple[str, Any]]:
        """Emit content as received, then a fully validated final NativeResult.

        A consumer must buffer the content if tools are available, as a tool
        decision can arrive after initial content in native SSE.
        """
        payload = _prepare_payload(messages, registry, tools, self.model, temperature, max_tokens, stream=True)
        accumulator = _Accumulator(registry)
        try:
            with self._client() as client:
                with client.stream("POST", API_URL, json=payload) as response:
                    _check_http_status(response)
                    bytes_seen = 0
                    data_lines: list[str] = []
                    done = False
                    for line in response.iter_lines():
                        bytes_seen += len(line.encode("utf-8"))
                        if bytes_seen > MAX_NATIVE_STREAM:
                            raise ModelProtocolError("Native SSE exceeds size limit")
                        if line == "":
                            if not data_lines:
                                continue
                            data = "\n".join(data_lines)
                            data_lines.clear()
                            if data == "[DONE]":
                                done = True
                                break
                            for content in accumulator.add(_decode_json(data)):
                                yield ("text", content)
                        elif line.startswith("data:"):
                            data_lines.append(line[5:].lstrip(" "))
                        # Other fields (event, id, comments) are ignored.
                    if not done and data_lines:
                        data = "\n".join(data_lines)
                        if data == "[DONE]":
                            done = True
                        else:
                            for content in accumulator.add(_decode_json(data)):
                                yield ("text", content)
                    if not done:
                        raise ModelProtocolError("Native SSE ended without DONE")
        except httpx.RequestError as exc:
            raise NativeFailure("Native streaming transport error") from exc
        yield ("finish", accumulator.finish())
