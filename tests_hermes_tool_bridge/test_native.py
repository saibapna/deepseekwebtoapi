"""Native wire-level regression with httpx.MockTransport (no live tokens)."""
import json
import httpx
import pytest

from hermes_tool_bridge.contract import ToolRegistry, ModelProtocolError
from hermes_tool_bridge.native import (
    NativeDeepSeekBackend, NativeResult, NativeFailure, NativeUnsupported,
    NativeUnavailable,
)


def backend(handler, model="deepseek-v4-pro"):
    return NativeDeepSeekBackend(api_key="native-test-secret",
                                 transport=httpx.MockTransport(handler), model=model)


def registry(tools, choice="auto", parallel=True):
    return ToolRegistry.from_request(tools, choice, parallel)


def native_tool(name, arguments, id="native_123"):
    return {"id": id, "type": "function", "function": {
        "name": name, "arguments": arguments}}


def answer(message=None, finish="stop"):
    return {"choices": [{"index": 0, "finish_reason": finish,
                         "message": message or {"role": "assistant", "content": "Hello"}}]}


def sse(*events):
    return "".join("data: " + json.dumps(x) + "\n\n" for x in events) + "data: [DONE]\n\n"


def event(delta=None, finish=None):
    return {"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]}


def test_native_json_tool_and_request_is_scoped(tools):
    def handler(req):
        assert req.url.host == "api.deepseek.com"
        assert req.headers["Authorization"] == "Bearer native-test-secret"
        payload = json.loads(req.content)
        assert payload["model"] == "deepseek-v4-pro"
        assert payload["thinking"] == {"type":"disabled"}
        assert payload["tools"][0]["function"]["name"] == "read_file"
        assert payload["parallel_tool_calls"] is True
        return httpx.Response(200, json=answer({"role":"assistant","content":None,"tool_calls":[
            native_tool("read_file", '{"path":"notes.txt"}')]}, "tool_calls"))
    result=backend(handler).complete([{"role":"user","content":"read notes"}], registry(tools), tools, "deepseek-chat")
    assert isinstance(result, NativeResult)
    assert result.content is None
    assert result.calls[0]["id"].startswith("call_")
    assert result.calls[0]["id"] != "native_123"
    assert json.loads(result.calls[0]["function"]["arguments"]) == {"path": "notes.txt"}


def test_native_json_text_no_tools():
    def handler(req):
        assert "tools" not in json.loads(req.content)
        return httpx.Response(200, json=answer())
    out=backend(handler).complete([{"role":"user","content":"hello"}],registry([]),[],"deepseek-chat")
    assert out == NativeResult(content="Hello", calls=[])


@pytest.mark.parametrize("raw",[
    '{"path":"ok","path":"evil"}',
    '{"path":"x","hidden":123}',
    '{"path":NaN}',
    '{"path":false}',
    '{"path":',
])
def test_native_rejects_invalid_arguments(tools, raw):
    def handler(_):
        return httpx.Response(200,json=answer({"role":"assistant","content":None,"tool_calls":[
            native_tool("read_file",raw)]},"tool_calls"))
    with pytest.raises(ModelProtocolError):
        backend(handler).complete([{"role":"user","content":"x"}],registry(tools),tools,"deepseek-chat")


def test_native_required_and_named_choice(tools):
    def handler(_): return httpx.Response(200,json=answer())
    with pytest.raises(ModelProtocolError):
        backend(handler).complete([{"role":"user","content":"x"}],registry(tools,"required"),tools,"deepseek-chat")


def test_native_unsupported_detected_400_explicit(tools):
    def handler(_):
        return httpx.Response(400,json={"error":{"message":"tool calling not supported by this model"}})
    with pytest.raises(NativeUnsupported):
        backend(handler).complete([{"role":"user","content":"x"}],registry(tools),tools,"deepseek-chat")


@pytest.mark.parametrize("code,message",[(401,"tool calling not supported"),(500,"tool calling not supported"),
                                             (400,"invalid prompt"),(429,"tool calling not supported")])
def test_native_other_failures_never_fallback(code,message,tools):
    def handler(_): return httpx.Response(code,json={"error":{"message":message}})
    with pytest.raises(NativeFailure):
        backend(handler).complete([{"role":"user","content":"x"}],registry(tools),tools,"deepseek-chat")


def test_native_timeout_fails_without_fallback(tools):
    def handler(_): raise httpx.ReadTimeout("hidden-upstream-secret")
    with pytest.raises(NativeFailure) as err:
        backend(handler).complete([{"role":"user","content":"x"}],registry(tools),tools,"deepseek-chat")
    assert "hidden-upstream-secret" not in str(err.value)


def test_native_unconfigured():
    p=NativeDeepSeekBackend(api_key="")
    assert not p.configured
    with pytest.raises(NativeUnavailable):
        p.complete([{"role":"user","content":"x"}],registry([]),[],"deepseek-chat")


def test_native_tool_names_incompatible_do_not_reach_upstream():
    called=[]
    tools=[{"type":"function","function":{"name":"a.b","parameters":{"type":"object"}}}]
    def handler(_): called.append(True); return httpx.Response(200,json=answer())
    with pytest.raises(NativeUnsupported):
        backend(handler).complete([{"role":"user","content":"x"}],registry(tools),tools,"deepseek-chat")
    assert called == []


def test_native_sse_plain_incremental():
    text=sse(event({"role":"assistant"}),event({"content":"a"}),
             event({"content":"bc"}),event({},"stop"))
    b=backend(lambda req: httpx.Response(200,text=text,headers={"content-type":"text/event-stream"}))
    events=list(b.stream_reply([{"role":"user","content":"x"}],registry([]),[],"deepseek-chat"))
    assert events[:2] == [("text","a"),("text","bc")]
    assert events[-1] == ("finish",NativeResult("abc",[]))


def test_native_sse_fragmented_tool_delta(tools):
    raw=sse(
        event({"tool_calls":[{"index":0,"id":"n-1","type":"function","function":{"name":"read_file","arguments":"{\\\"path\\\":\\\""}}]}),
        event({"tool_calls":[{"index":0,"function":{"arguments":"foo.txt\\\"}"}}]}),
        event({},"tool_calls"),
    )
    b=backend(lambda req:httpx.Response(200,text=raw,headers={"content-type":"text/event-stream"}))
    with pytest.raises(ModelProtocolError):
        # Deliberately malformed escaped JSON must fail closed.
        list(b.stream_reply([{"role":"user","content":"x"}],registry(tools),tools,"deepseek-chat"))


def test_native_sse_valid_tool_fragments(tools):
    raw=sse(event({"tool_calls":[{"index":0,"id":"n-1","type":"function","function":{
        "name":"read_file","arguments":'{"path":"'}}]}),
        event({"tool_calls":[{"index":0,"function":{"arguments":"foo.txt\"}"}}]}),
        event({},"tool_calls"))
    b=backend(lambda req: httpx.Response(200,text=raw))
    events=list(b.stream_reply([{"role":"user","content":"x"}],registry(tools),tools,"deepseek-chat"))
    assert len(events)==1 and events[0][0]=="finish"
    assert json.loads(events[0][1].calls[0]["function"]["arguments"]) == {"path":"foo.txt"}


def test_native_sse_missing_done_is_protocol_error():
    raw='data: '+json.dumps(event({"content":"partial"}))+'\n\n'
    b=backend(lambda req: httpx.Response(200,text=raw))
    it=b.stream_reply([{"role":"user","content":"x"}],registry([]),[],"deepseek-chat")
    assert next(it)==("text","partial")
    with pytest.raises(ModelProtocolError): list(it)


def test_native_sse_wrong_finish_reason():
    raw=sse(event({"content":"x"}),event({},"length"))
    b=backend(lambda req: httpx.Response(200,text=raw))
    with pytest.raises(ModelProtocolError):
        list(b.stream_reply([{"role":"user","content":"x"}],registry([]),[],"deepseek-chat"))


def test_native_tool_result_roundtrip(tools):
    previous={"role":"assistant","content":None,"tool_calls":[native_tool("read_file",'{"path":"note.txt"}',id="call_hermes123")]}
    messages=[{"role":"user","content":"read"},previous,
              {"role":"tool","tool_call_id":"call_hermes123","content":"file text"}]
    def handler(req):
        got=json.loads(req.content)["messages"]
        assert got[-1]["tool_call_id"]=="call_hermes123"
        assert got[-2]["tool_calls"][0]["id"]=="call_hermes123"
        return httpx.Response(200,json=answer())
    assert backend(handler).complete(messages,registry(tools),tools,"deepseek-chat").content=="Hello"
