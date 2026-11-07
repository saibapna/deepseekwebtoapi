"""End-to-end mocked Hermes/native/web conversation routing and SSE tests."""
import json

import pytest
from fastapi.testclient import TestClient

from hermes_tool_bridge.app import create_app
from hermes_tool_bridge.native import NativeResult, NativeUnsupported, NativeFailure
from hermes_tool_bridge.contract import ToolRegistry
from .conftest import chat_request, sse_frames, sse_json

HEADERS={"Authorization":"Bearer test-key"}


class Web:
    def __init__(self, fragments, recorded):
        self.fragments=fragments
        self.recorded=recorded
    def iter_text(self, prompt):
        self.recorded.append(prompt)
        yield from self.fragments


class Native:
    configured=True
    def __init__(self, reply=None, error=None, stream_events=None):
        self.reply=reply or NativeResult("native answer",[])
        self.error=error
        self.stream_events=stream_events
        self.calls=0
    def complete(self,*args):
        self.calls+=1
        if self.error: raise self.error
        return self.reply
    def stream_reply(self,*args):
        self.calls+=1
        if self.error: raise self.error
        if self.stream_events is not None:
            yield from self.stream_events
        else:
            if self.reply.content:
                yield ("text",self.reply.content)
            yield ("finish",self.reply)


def client(mode, native, web_response="web answer", fallback=False):
    received=[]
    app=create_app(
        backend_factory=lambda thinking,search: Web([web_response],received),
        native_factory=lambda: native, api_key="test-key",tool_mode=mode,
        allow_web_fallback=fallback,heartbeat_seconds=.02,
    )
    return TestClient(app), received


def basic(**kwargs):
    return {"model":"deepseek-chat","messages":[{"role":"user","content":"Hi"}],**kwargs}


def test_bridge_default_ignores_native():
    native=Native()
    c,seen=client("bridge",native)
    r=c.post("/v1/chat/completions",headers=HEADERS,json=basic())
    assert r.status_code==200
    assert r.json()["choices"][0]["message"]["content"]=="web answer"
    assert native.calls==0 and len(seen)==1


def test_hybrid_native_first_nonstream():
    native=Native()
    c,seen=client("hybrid",native,fallback=True)
    r=c.post("/v1/chat/completions",headers=HEADERS,json=basic())
    assert r.status_code==200
    assert r.json()["choices"][0]["message"]["content"]=="native answer"
    assert native.calls==1 and seen==[]


def test_hybrid_fallback_only_on_explicit_unsupported():
    native=Native(error=NativeUnsupported("no tool calling"))
    c,seen=client("hybrid",native,fallback=True)
    r=c.post("/v1/chat/completions",headers=HEADERS,json=basic())
    assert r.status_code==200
    assert r.json()["choices"][0]["message"]["content"]=="web answer"
    assert native.calls==1 and len(seen)==1


def test_hybrid_fallback_denied_by_default():
    native=Native(error=NativeUnsupported("no tools"))
    c,seen=client("hybrid",native,fallback=False)
    r=c.post("/v1/chat/completions",headers=HEADERS,json=basic())
    assert r.status_code==502 and r.json()["error"]["type"]=="native_unsupported"
    assert seen==[]


def test_native_failure_never_fallback_and_is_sanitized():
    native=Native(error=NativeFailure("my-secret-token"))
    c,seen=client("hybrid",native,fallback=True)
    r=c.post("/v1/chat/completions",headers=HEADERS,json=basic())
    assert r.status_code==502 and "my-secret-token" not in r.text
    assert seen==[]


def test_unconfigured_native_hybrid_optin_and_native_failclosed():
    native=Native()
    native.configured=False
    c,seen=client("hybrid",native,fallback=True)
    assert c.post("/v1/chat/completions",headers=HEADERS,json=basic()).status_code==200
    assert len(seen)==1 and native.calls==0
    c,seen=client("native",native,fallback=True)
    assert c.post("/v1/chat/completions",headers=HEADERS,json=basic()).status_code==503
    assert seen==[]


def test_hybrid_web_search_has_explicit_route():
    native=Native()
    c,seen=client("hybrid",native,fallback=True)
    r=c.post("/v1/chat/completions",headers=HEADERS,json=basic(search_enabled=True))
    assert r.status_code==200 and len(seen)==1 and native.calls==0
    c,seen=client("hybrid",native,fallback=False)
    assert c.post("/v1/chat/completions",headers=HEADERS,json=basic(search_enabled=True)).status_code==422
    c,seen=client("native",native,fallback=True)
    assert c.post("/v1/chat/completions",headers=HEADERS,json=basic(search_enabled=True)).status_code==422


def test_native_validated_tool_is_returned_as_openai_toolcall(tools):
    registry=ToolRegistry.from_request(tools,"auto",True)
    _,calls=registry.validate_response('<hermes_tool_call>{"version":"1","tool_calls":[{"name":"read_file","arguments":{"path":"file.txt"}}]}</hermes_tool_call>')
    native=Native(reply=NativeResult(None,calls))
    c,_=client("hybrid",native,fallback=True)
    r=c.post("/v1/chat/completions",headers=HEADERS,json=chat_request(tools))
    assert r.status_code==200
    data=r.json()["choices"][0]
    assert data["finish_reason"]=="tool_calls"
    assert data["message"]["tool_calls"][0]["function"]["name"]=="read_file"


def test_hybrid_stream_native_text_incremental():
    native=Native(stream_events=[("text","a"),("text","b"),("finish",NativeResult("ab",[]))])
    c,_=client("hybrid",native,fallback=True)
    r=c.post("/v1/chat/completions",headers=HEADERS,json=basic(stream=True))
    assert sse_frames(r)[-1]=="[DONE]"
    deltas=[chunk["choices"][0]["delta"].get("content","") for chunk in sse_json(r)]
    assert [x for x in deltas if x]==["a","b"]


def test_hybrid_stream_native_tools_are_validated_before_sse(tools):
    registry=ToolRegistry.from_request(tools,"auto",True)
    _,calls=registry.validate_response('<hermes_tool_call>{"version":"1","tool_calls":[{"name":"read_file","arguments":{"path":"file.txt"}}]}</hermes_tool_call>')
    native=Native(stream_events=[("text","internal text"),("finish",NativeResult(None,calls))])
    c,_=client("hybrid",native,fallback=False)
    r=c.post("/v1/chat/completions",headers=HEADERS,json=chat_request(tools,stream=True))
    assert r.status_code==200 and sse_frames(r)[-1]=="[DONE]"
    chunks=sse_json(r)
    assert not any(x["choices"][0]["delta"].get("content") for x in chunks)
    assert any("tool_calls" in x["choices"][0]["delta"] for x in chunks)
    assert chunks[-1]["choices"][0]["finish_reason"]=="tool_calls"


def test_hybrid_stream_fallback_after_unsupported(tools,envelope):
    native=Native(error=NativeUnsupported("no tools"))
    c,seen=client("hybrid",native,web_response=envelope,fallback=True)
    r=c.post("/v1/chat/completions",headers=HEADERS,json=chat_request(tools,stream=True))
    assert r.status_code==200 and sse_frames(r)[-1]=="[DONE]"
    assert len(seen)==1 and native.calls==1
    assert any("tool_calls" in x["choices"][0]["delta"] for x in sse_json(r))


def test_hybrid_stream_rejects_late_native_unsupported_without_fallback():
    class LateNative(Native):
        def stream_reply(self,*args):
            yield ("text","already sent")
            raise NativeUnsupported("late unexpected")
    c,seen=client("hybrid",LateNative(),fallback=True)
    r=c.post("/v1/chat/completions",headers=HEADERS,json=basic(stream=True))
    assert sse_frames(r)[-1] != "[DONE]"
    assert any(x.get("error",{}).get("type")=="native_unsupported" for x in sse_json(r))
    assert seen==[]


def test_native_stream_error_doesnt_mark_done():
    class Broken(Native):
        def stream_reply(self,*args):
            yield ("text","some output")
            raise NativeFailure("credential")
    c,seen=client("native",Broken(),fallback=True)
    r=c.post("/v1/chat/completions",headers=HEADERS,json=basic(stream=True))
    assert "credential" not in r.text
    assert sse_frames(r)[-1] != "[DONE]"
    assert seen==[]


def test_hybrid_multi_turn_reuses_tool_result_id(tools):
    registry=ToolRegistry.from_request(tools,"auto",True)
    _,calls=registry.validate_response('<hermes_tool_call>{"version":"1","tool_calls":[{"name":"calculator","arguments":{"a":2,"b":3}}]}</hermes_tool_call>')
    class Recorder(Native):
        def complete(self,messages,*args):
            self.calls+=1
            if self.calls==1:
                return NativeResult(None,calls)
            assert messages[-1]["role"]=="tool"
            assert messages[-1]["tool_call_id"]==calls[0]["id"]
            return NativeResult("5",[])
    recorder=Recorder()
    c,_=client("hybrid",recorder)
    req=chat_request(tools)
    first=c.post("/v1/chat/completions",headers=HEADERS,json=req).json()["choices"][0]["message"]
    req["messages"].append(first)
    req["messages"].append({"role":"tool","tool_call_id":first["tool_calls"][0]["id"],"content":"5"})
    second=c.post("/v1/chat/completions",headers=HEADERS,json=req)
    assert second.status_code==200 and second.json()["choices"][0]["message"]["content"]=="5"


def test_invalid_mode_fails_at_startup():
    with pytest.raises(ValueError): create_app(tool_mode="magical")
