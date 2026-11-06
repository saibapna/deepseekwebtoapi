"""Adapter for the existing dsk.api.DeepSeekAPI streaming generator.

No browser/Cloudflare bypass services are started by the bridge. The original
library must be available in the same Python environment, and upstream timeouts
should be installed with the bundled optional stability script.
"""
from __future__ import annotations

import os
from collections.abc import Iterator


class UpstreamUnavailable(RuntimeError):
    """Sanitized error type, never contains upstream response payloads."""


class DeepSeekWebBackend:
    def __init__(self, *, token: str | None = None, thinking: bool = False, search: bool = False):
        self.token = token or os.environ.get("DEEPSEEK_AUTH_TOKEN")
        self.thinking = thinking
        self.search = search

    def iter_text(self, prompt: str) -> Iterator[str]:
        if not self.token:
            raise UpstreamUnavailable("DEEPSEEK_AUTH_TOKEN is missing")
        try:
            from dsk.api import DeepSeekAPI
        except ImportError as exc:
            raise UpstreamUnavailable("Original dsk package is unavailable") from exc
        # Deliberately create a fresh ephemeral web session for every OpenAI
        # request. Hermes supplies prior messages, including tool results, and
        # the bridge replays those messages in the prompt. No cross-user state.
        api = DeepSeekAPI(self.token)
        session_id = api.create_chat_session()
        response = api.chat_completion(
            session_id, prompt, thinking_enabled=self.thinking, search_enabled=self.search
        )
        try:
            for chunk in response:
                if not isinstance(chunk, dict):
                    continue
                if chunk.get("finish_reason"):
                    break
                if chunk.get("type") == "text" and isinstance(chunk.get("content"), str):
                    if chunk["content"]:
                        yield chunk["content"]
                # Thinking / internal events are intentionally not exposed as
                # content or function arguments.
        finally:
            # With the optional dsk stability patch this also closes its HTTP
            # response. Unpatched dsk lacks a response.close finally block.
            close = getattr(response, "close", None)
            if callable(close):
                close()
