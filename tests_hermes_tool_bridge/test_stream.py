import json
import time
import threading
import pytest
from hermes_tool_bridge.contract import OPEN_TAG, CLOSE_TAG, ToolRegistry, ModelProtocolError
from hermes_tool_bridge.stream import ReplyClassifier, StreamPump, collect_reply, completion_identity, stream_frames


def registry(tools, choice="auto"):
    return ToolRegistry.from_request(tools,choice,True)


def parse(frames):
    return [json.loads(x[6:]) for x in frames if x.startswith('data: ') and x.strip() != 'data: [DONE]']


def test_live_plain_text_streams_without_full_buffer(tools):
    c=ReplyClassifier(registry(tools))
    assert c.push('He') == ['He']
    assert c.push('llo') == ['llo']
    assert c.finish() == ([],[])


def test_tool_prefix_not_leaked(tools, envelope):
    c=ReplyClassifier(registry(tools))
    output=[]
    for part in [envelope[:5],envelope[5:17],envelope[17:50],envelope[50:]]:
        output.extend(c.push(part))
    assert output == []
    tail,calls=c.finish()
    assert tail==[] and calls[0]['function']['name']=='read_file'


def test_partial_tag_rejected(tools):
    c=ReplyClassifier(registry(tools)); c.push(' <hermes_tool_cal')
    with pytest.raises(ModelProtocolError): c.finish()


def test_response_too_long_does_not_allocate_forever(tools):
    c=ReplyClassifier(registry(tools))
    with pytest.raises(ModelProtocolError): c.push('x'*160001)


def test_chunked_tool_stream_has_delta_index_and_finish(tools,envelope):
    p=StreamPump(lambda: (envelope[i:i+4] for i in range(0,len(envelope),4)),lambda: None,heartbeat=.2).start()
    frames=list(stream_frames(p,registry(tools),completion_identity('deepseek-chat')))
    parsed=parse(frames)
    assert frames[-1]=='data: [DONE]\n\n'
    assert parsed[0]['choices'][0]['delta']=={'role':'assistant'}
    calls=[x['choices'][0]['delta']['tool_calls'][0] for x in parsed if x['choices'][0]['delta'].get('tool_calls')]
    assert calls[0]['index']==0 and calls[0]['id'].startswith('call_')
    assert ''.join(c.get('function',{}).get('arguments','') for c in calls)=='{"path":"notes.txt"}'
    assert parsed[-1]['choices'][0]['finish_reason']=='tool_calls'


def test_plain_stream_sends_token_deltas(tools):
    p=StreamPump(lambda: iter(['Hel','lo','!']),lambda: None,heartbeat=.2).start()
    frames=list(stream_frames(p,registry(tools),completion_identity('deepseek-chat')))
    parsed=parse(frames)
    texts=[x['choices'][0]['delta']['content'] for x in parsed if 'content' in x['choices'][0]['delta']]
    assert texts==['Hel','lo','!']
    assert parsed[-1]['choices'][0]['finish_reason']=='stop'


def test_invalid_tool_stream_fails_closed(tools):
    raw='<hermes_tool_call>{"version":"1","tool_calls":[{"name":"shell","arguments":{}}]}</hermes_tool_call>'
    p=StreamPump(lambda: iter([raw]),lambda:None,heartbeat=.2).start()
    frames=list(stream_frames(p,registry(tools),completion_identity('deepseek-chat')))
    assert not any('"tool_calls"' in f and '"delta"' in f for f in frames)
    assert any('tool_protocol_error' in f for f in frames)
    assert 'data: [DONE]' not in ''.join(frames)


def test_stream_worker_exception_does_not_leak(tools):
    def source():
        yield 'Ok'
        raise RuntimeError('very-secret-token')
    p=StreamPump(source,lambda:None,heartbeat=.2).start()
    frames=list(stream_frames(p,registry(tools),completion_identity('deepseek-chat')))
    assert 'upstream_error' in ''.join(frames)
    assert 'very-secret-token' not in ''.join(frames)


def test_heartbeat_during_slow_upstream(tools):
    def source():
        time.sleep(.06)
        yield 'hello'
    p=StreamPump(source,lambda:None,heartbeat=.01).start()
    frames=list(stream_frames(p,registry(tools),completion_identity('deepseek-chat')))
    assert ': keepalive\n\n' in frames


def test_stream_cancel_signals_stop(tools):
    entered=threading.Event(); released=threading.Event()
    def source():
        for i in range(100):
            entered.set()
            yield 'a'
        released.set()
    p=StreamPump(source,lambda:None,heartbeat=.1).start()
    gen=stream_frames(p,registry(tools),completion_identity('deepseek-chat'))
    next(gen)
    gen.close()
    assert p._stop.is_set()


def test_collect_response_closes_generator(tools):
    closed=[]
    def source():
        try: yield 'text'
        finally: closed.append(True)
    assert collect_reply(source(),registry(tools))==('text',[])
    assert closed == [True]
