import json
import pytest
from fastapi.testclient import TestClient
from hermes_tool_bridge.app import create_app


@pytest.fixture
def tools():
    return [
        {"type": "function", "function": {
            "name": "read_file",
            "description": "Read a file, after Hermes checks permissions",
            "parameters": {"type": "object", "properties": {"path": {"type": "string", "minLength": 1}}, "required": ["path"]},
        }},
        {"type": "function", "function": {
            "name": "calculator",
            "parameters": {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]},
        }},
    ]


@pytest.fixture
def envelope():
    return '<hermes_tool_call>{"version":"1","tool_calls":[{"name":"read_file","arguments":{"path":"notes.txt"}}]}</hermes_tool_call>'


class FakeBackend:
    def __init__(self, fragments, prompts):
        self.fragments = fragments
        self.prompts = prompts

    def iter_text(self, prompt):
        self.prompts.append(prompt)
        for item in self.fragments:
            if isinstance(item, Exception):
                raise item
            yield item


@pytest.fixture
def client_factory():
    def make(fragments, **opts):
        prompts = []
        backend = lambda think, search: FakeBackend(fragments, prompts)
        app = create_app(backend_factory=backend, api_key="unit-test-secret", **opts)
        return TestClient(app), prompts
    return make


def chat_request(tools, **kwargs):
    return {"model": "deepseek-chat", "messages": [{"role": "user", "content": "read notes"}], "tools": tools, **kwargs}


def sse_frames(response):
    return [line[6:] for line in response.text.splitlines() if line.startswith('data: ')]


def sse_json(response):
    return [json.loads(line) for line in sse_frames(response) if line != '[DONE]']
