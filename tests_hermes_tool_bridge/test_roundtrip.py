"""Contract-level Hermes roundtrip: a fake executor lives outside the bridge."""
import json

from fastapi.testclient import TestClient

from hermes_tool_bridge.app import create_app
from .conftest import chat_request, sse_json


class CycleBackend:
    def __init__(self, transcript):
        self.transcript = transcript

    def iter_text(self, prompt):
        self.transcript.append(prompt)
        if '"role":"tool"' in prompt:
            yield 'Hermes reported that the total is 5.'
        else:
            yield '<hermes_tool_call>{"version":"1","tool_calls":[{"name":"calculator","arguments":{"a":2,"b":3}}]}</hermes_tool_call>'


def test_hermes_executes_after_tool_call_not_bridge(tools):
    observed=[]
    app=create_app(backend_factory=lambda t,s:CycleBackend(observed),api_key='secret')
    http=TestClient(app)
    headers={'Authorization':'Bearer secret'}
    one=http.post('/v1/chat/completions',headers=headers,json=chat_request(tools,stream=False))
    assert one.status_code==200
    first=one.json()['choices'][0]['message']
    assert first['content'] is None
    # Fake Hermes validates and runs a harmless tool after receiving the
    # response. The bridge has no authority or callback to execute anything.
    call=first['tool_calls'][0]
    args=json.loads(call['function']['arguments'])
    fake_hermes_result=str(args['a']+args['b'])
    messages=[
        {'role':'user','content':'calculate 2 + 3'},
        first,
        {'role':'tool','tool_call_id':call['id'],'content':fake_hermes_result},
    ]
    two=http.post('/v1/chat/completions',headers=headers,json={
        'model':'deepseek-chat','tools':tools,'messages':messages,'stream':True,
    })
    assert two.status_code==200
    joined=''.join(x['choices'][0]['delta'].get('content','') for x in sse_json(two))
    assert joined=='Hermes reported that the total is 5.'
    assert call['id'] in observed[-1]
    assert '"content":"5"' in observed[-1]


def test_streaming_call_arguments_assemble(tools):
    envelope='<hermes_tool_call>{"version":"1","tool_calls":[{"name":"calculator","arguments":{"a":10,"b":12}}]}</hermes_tool_call>'
    class StreamingBackend:
        def iter_text(self,prompt):
            for char in envelope: yield char
    app=create_app(backend_factory=lambda t,s:StreamingBackend(),api_key='secret')
    response=TestClient(app).post('/v1/chat/completions',headers={'Authorization':'Bearer secret'},json=chat_request(tools,stream=True))
    frames=sse_json(response)
    deltas=[x['choices'][0]['delta'].get('tool_calls',[]) for x in frames]
    calls=[delta[0] for delta in deltas if delta]
    assert calls[0]['function']['name']=='calculator'
    args=''.join(x.get('function',{}).get('arguments','') for x in calls)
    assert json.loads(args)=={'a':10,'b':12}
    assert frames[-1]['choices'][0]['finish_reason']=='tool_calls'
