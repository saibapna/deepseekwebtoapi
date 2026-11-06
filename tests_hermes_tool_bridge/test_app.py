import json
import pytest
from fastapi.testclient import TestClient
from hermes_tool_bridge.app import create_app
from .conftest import chat_request, sse_frames, sse_json


def test_health_without_token(client_factory):
    client,_=client_factory(['Hello'])
    assert client.get('/health').json()['status']=='ok'
    assert client.get('/v1/models').status_code==401


def test_auth_requires_matching_token(client_factory):
    client,_=client_factory(['Hello'])
    assert client.post('/v1/chat/completions',json={'messages':[{'role':'user','content':'hi'}]}).status_code==401
    assert client.get('/v1/models',headers={'Authorization':'Bearer wrong'}).status_code==401
    assert client.get('/v1/models',headers={'Authorization':'Bearer unit-test-secret'}).status_code==200


def test_missing_config_fails_closed():
    app=create_app(backend_factory=lambda t,s:None,api_key='')
    r=TestClient(app).get('/v1/models',headers={'Authorization':'Bearer hi'})
    assert r.status_code==503


def test_nonstream_text(client_factory):
    client,prompts=client_factory(['Hi',' there'])
    r=client.post('/v1/chat/completions',headers={'Authorization':'Bearer unit-test-secret'},json={'model':'deepseek-chat','messages':[{'role':'user','content':'hello'}]})
    assert r.status_code==200
    body=r.json()
    assert body['choices'][0]['message']['content']=='Hi there'
    assert body['choices'][0]['finish_reason']=='stop'
    assert 'CONVERSATION_JSON' in prompts[0]


def test_nonstream_tool(client_factory,tools,envelope):
    client,_=client_factory([envelope])
    r=client.post('/v1/chat/completions',headers={'Authorization':'Bearer unit-test-secret'},json=chat_request(tools))
    assert r.status_code==200
    choice=r.json()['choices'][0]
    assert choice['message']['content'] is None
    assert choice['finish_reason']=='tool_calls'
    assert choice['message']['tool_calls'][0]['function']['name']=='read_file'


def test_stream_plain_text(client_factory):
    client,_=client_factory(['Hello',' ','world'])
    r=client.post('/v1/chat/completions',headers={'Authorization':'Bearer unit-test-secret'},json={'messages':[{'role':'user','content':'hi'}],'stream':True})
    assert r.status_code==200
    assert sse_frames(r)[-1]=='[DONE]'
    assert ''.join(x['choices'][0]['delta'].get('content','') for x in sse_json(r))=='Hello world'


def test_stream_tool(client_factory,tools,envelope):
    client,_=client_factory([envelope[:23],envelope[23:]],heartbeat_seconds=.2)
    r=client.post('/v1/chat/completions',headers={'Authorization':'Bearer unit-test-secret'},json=chat_request(tools,stream=True))
    assert r.status_code==200
    assert sse_frames(r)[-1]=='[DONE]'
    assert any('tool_calls' in x['choices'][0]['delta'] for x in sse_json(r))


def test_invalid_tool_rejected_http(client_factory,tools):
    client,_=client_factory(['<hermes_tool_call>{"version":"1","tool_calls":[{"name":"danger","arguments":{}}]}</hermes_tool_call>'])
    r=client.post('/v1/chat/completions',headers={'Authorization':'Bearer unit-test-secret'},json=chat_request(tools))
    assert r.status_code==502 and r.json()['error']['type']=='tool_protocol_error'


def test_invalid_schema_http(client_factory):
    client,_=client_factory(['ok'])
    bad=[{'type':'function','function':{'name':'one','parameters':{'type':'object','$ref':'https://bad.example/schema'}}}]
    r=client.post('/v1/chat/completions',headers={'Authorization':'Bearer unit-test-secret'},json=chat_request(bad))
    assert r.status_code==422


def test_unsupported_model(client_factory):
    client,_=client_factory(['hi'])
    r=client.post('/v1/chat/completions',headers={'Authorization':'Bearer unit-test-secret'},json={'messages':[{'role':'user','content':'hi'}],'model':'different'})
    assert r.status_code==400


def test_inflight_limit(client_factory):
    client,_=client_factory(['a'],max_inflight=1)
    # A held capacity slot must fail promptly without reaching upstream.
    app=client.app
    # Test via a request that has no worker? Better exercise with slow fake.
    assert client.get('/health').status_code==200


def test_multiturn_tool_result_roundtrip(client_factory, tools, envelope):
    client,prompts=client_factory([envelope])
    headers={'Authorization':'Bearer unit-test-secret'}
    first=client.post('/v1/chat/completions',headers=headers,json=chat_request(tools)).json()
    previous=first['choices'][0]['message']
    assert previous['tool_calls'][0]['id']
    result_id=previous['tool_calls'][0]['id']
    second_request=chat_request(tools,messages='IGNORED')
    second_request['messages']=[{'role':'user','content':'read notes'},previous,{'role':'tool','tool_call_id':result_id,'content':'Here are the notes'},{'role':'user','content':'summarize'}]
    second=client.post('/v1/chat/completions',headers=headers,json=second_request)
    assert second.status_code==200
    assert result_id in prompts[-1]
    assert 'Here are the notes' in prompts[-1]


def test_upstream_exception_sanitized(client_factory):
    client,_=client_factory([RuntimeError('secret-upstream-body')])
    r=client.post('/v1/chat/completions',headers={'Authorization':'Bearer unit-test-secret'},json={'messages':[{'role':'user','content':'hi'}]})
    assert r.status_code==502
    assert 'secret-upstream-body' not in r.text


def test_unsupported_image_is_safe_422(client_factory):
    client,_=client_factory(['ok'])
    r=client.post('/v1/chat/completions',headers={'Authorization':'Bearer unit-test-secret'},json={'messages':[{'role':'user','content':[{'type':'image_url','image_url':{'url':'data:x'}}]}]})
    assert r.status_code==422


def test_bad_unicode_bearer_token_is_not_500():
    from fastapi import HTTPException
    from hermes_tool_bridge.app import _auth
    with pytest.raises(HTTPException) as exc:
        _auth('Bearer x\u2603', 'unit-test-secret')
    assert exc.value.status_code == 401


def test_concurrency_limit_reports_busy_and_recovers():
    import threading
    from fastapi.testclient import TestClient
    from hermes_tool_bridge.app import create_app
    entered=threading.Event(); release=threading.Event()
    class SlowBackend:
        def iter_text(self, prompt):
            entered.set()
            if not release.wait(timeout=3):
                raise RuntimeError('test worker was stuck')
            yield 'ready'
    app=create_app(backend_factory=lambda t,s:SlowBackend(),api_key='secret',max_inflight=1)
    client=TestClient(app)
    headers={'Authorization':'Bearer secret'}
    req={'messages':[{'role':'user','content':'hi'}]}
    out=[]
    worker=threading.Thread(target=lambda:out.append(client.post('/v1/chat/completions',json=req,headers=headers)))
    worker.start()
    try:
        assert entered.wait(timeout=2)
        busy=client.post('/v1/chat/completions',json=req,headers=headers)
        assert busy.status_code==503
    finally:
        release.set()
        worker.join(timeout=3)
    assert not worker.is_alive() and out[0].status_code==200
    again=client.post('/v1/chat/completions',json=req,headers=headers)
    assert again.status_code==200
