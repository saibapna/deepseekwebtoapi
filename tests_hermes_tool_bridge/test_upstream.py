import sys
import types
import pytest
from hermes_tool_bridge.upstream import DeepSeekWebBackend, UpstreamUnavailable


def test_missing_token_fails_closed(monkeypatch):
    monkeypatch.delenv('DEEPSEEK_AUTH_TOKEN',raising=False)
    with pytest.raises(UpstreamUnavailable):
        list(DeepSeekWebBackend().iter_text('hi'))


def test_original_dsk_adapter_preserves_flags_and_finishes(monkeypatch):
    calls=[]
    closed=[]
    class API:
        def __init__(self,token): calls.append(('init',token))
        def create_chat_session(self): calls.append(('session',));return 'session-id'
        def chat_completion(self,session,prompt,**options):
            calls.append(('completion',session,prompt,options))
            def gen():
                try:
                    yield {'type':'thinking','content':'secret chain'}
                    yield {'type':'text','content':'answer'}
                    yield {'finish_reason':'stop'}
                finally: closed.append(True)
            return gen()
    module=types.ModuleType('dsk.api');module.DeepSeekAPI=API
    package=types.ModuleType('dsk');package.__path__=[]
    monkeypatch.setitem(sys.modules,'dsk',package)
    monkeypatch.setitem(sys.modules,'dsk.api',module)
    out=list(DeepSeekWebBackend(token='token',thinking=True,search=True).iter_text('prompt'))
    assert out == ['answer']
    assert closed == [True]
    assert calls[-1]==('completion','session-id','prompt',{'thinking_enabled':True,'search_enabled':True})


def test_original_dsk_adapter_closes_on_consumer_cancel(monkeypatch):
    closed=[]
    class API:
        def __init__(self,token): pass
        def create_chat_session(self): return 's'
        def chat_completion(self,*args,**kwargs):
            def gen():
                try:
                    yield {'type':'text','content':'first'}
                    yield {'type':'text','content':'second'}
                finally: closed.append(True)
            return gen()
    module=types.ModuleType('dsk.api');module.DeepSeekAPI=API
    package=types.ModuleType('dsk');package.__path__=[]
    monkeypatch.setitem(sys.modules,'dsk',package)
    monkeypatch.setitem(sys.modules,'dsk.api',module)
    gen=DeepSeekWebBackend(token='token').iter_text('prompt')
    assert next(gen)=='first'
    gen.close()
    assert closed == [True]
