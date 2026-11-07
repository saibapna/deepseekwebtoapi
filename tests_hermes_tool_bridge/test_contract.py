import json
import pytest
from jsonschema import ValidationError
from hermes_tool_bridge.contract import (
    OPEN_TAG, CLOSE_TAG, MAX_TOOLS, MAX_OUTPUT_BYTES,
    InvalidRequest, ModelProtocolError, ToolRegistry, make_prompt, normalize_messages,
)


def reg(tools, choice="auto", parallel=True):
    return ToolRegistry.from_request(tools, choice, parallel)


def call(name, args):
    return f'{OPEN_TAG}'+json.dumps({"version":"1","tool_calls":[{"name":name,"arguments":args}]})+f'{CLOSE_TAG}'


def test_valid_call_and_string_arguments(tools):
    content, calls = reg(tools).validate_response(call("read_file", {"path":"ok"}))
    assert content is None
    assert calls[0]["type"] == "function"
    assert calls[0]["id"].startswith("call_")
    assert json.loads(calls[0]["function"]["arguments"]) == {"path": "ok"}


def test_plain_text_is_not_a_call(tools):
    text = 'Hello. '+call("read_file", {"path":"x"})
    assert reg(tools).validate_response(text) == (text, [])


def test_fenced_json_is_not_a_call(tools):
    text = '```json\n'+call("read_file", {"path":"x"})+'\n```'
    assert reg(tools).validate_response(text) == (text, [])


def test_unknown_tool_rejected(tools):
    with pytest.raises(ModelProtocolError):
        reg(tools).validate_response(call("execute_shell", {"command":"rm -rf /"}))


def test_extra_argument_rejected(tools):
    with pytest.raises(ModelProtocolError):
        reg(tools).validate_response(call("read_file", {"path":"x", "hidden": 1}))


def test_missing_required_arg_rejected(tools):
    with pytest.raises(ModelProtocolError):
        reg(tools).validate_response(call("read_file", {}))


def test_bool_not_number(tools):
    with pytest.raises(ModelProtocolError):
        reg(tools).validate_response(call("calculator", {"a":True,"b":2}))


def test_invalid_json_rejected(tools):
    with pytest.raises(ModelProtocolError):
        reg(tools).validate_response(OPEN_TAG+'not-json'+CLOSE_TAG)


def test_duplicate_json_key_rejected(tools):
    raw = OPEN_TAG+'{"version":"1","version":"1","tool_calls":[]}'+CLOSE_TAG
    with pytest.raises(ModelProtocolError):
        reg(tools).validate_response(raw)


def test_nan_rejected(tools):
    raw = OPEN_TAG+'{"version":"1","tool_calls":[{"name":"calculator","arguments":{"a":NaN,"b":2}}]}'+CLOSE_TAG
    with pytest.raises(ModelProtocolError):
        reg(tools).validate_response(raw)


def test_tool_choice_required(tools):
    with pytest.raises(ModelProtocolError):
        reg(tools,"required").validate_response('No calls needed')
    assert reg(tools,"required").validate_response(call("read_file", {"path":"x"}))[1]


def test_tool_choice_named(tools):
    forced = {"type":"function","function":{"name":"calculator"}}
    with pytest.raises(ModelProtocolError):
        reg(tools,forced).validate_response(call("read_file", {"path":"x"}))
    assert reg(tools,forced).validate_response(call("calculator", {"a":1,"b":2}))[1]


def test_tool_choice_none_not_executed(tools):
    raw = call("read_file", {"path":"x"})
    assert reg(tools,"none").validate_response(raw) == (raw, [])


def test_missing_required_registry(tools):
    with pytest.raises(InvalidRequest): reg([],"required")
    with pytest.raises(InvalidRequest): reg(tools,{"type":"function","function":{"name":"not_registered"}})


def test_parallel_tool_batch(tools):
    data={"version":"1","tool_calls":[{"name":"read_file","arguments":{"path":"a"}},{"name":"calculator","arguments":{"a":1,"b":2}}]}
    raw=OPEN_TAG+json.dumps(data)+CLOSE_TAG
    assert len(reg(tools,parallel=True).validate_response(raw)[1]) == 2
    with pytest.raises(ModelProtocolError): reg(tools,parallel=False).validate_response(raw)


def test_oversized_calls_rejected(tools):
    with pytest.raises(ModelProtocolError): reg(tools).validate_response('x' * (MAX_OUTPUT_BYTES+1))
    with pytest.raises(ModelProtocolError): reg(tools).validate_response(call("read_file", {"path":"x"*17000}))


def test_duplicate_tools_and_bad_types(tools):
    with pytest.raises(InvalidRequest): reg(tools + tools)
    with pytest.raises(InvalidRequest): reg([{"type":"unhandled"}])
    with pytest.raises(InvalidRequest): reg(tools*49)


def test_schema_combinator_and_dynamic_properties(tools):
    typed=[{"type":"function","function":{"name":"choose","parameters":{
        "type":"object", "properties":{"mode":{"oneOf":[{"const":"fast"},{"const":"safe"}]},"metadata":{"type":"object","additionalProperties":{"type":"string"}}},
        "required":["mode"]}}}]
    accepted=call("choose",{"mode":"safe","metadata":{"env":"dev"}})
    assert reg(typed).validate_response(accepted)[1]
    with pytest.raises(ModelProtocolError): reg(typed).validate_response(call("choose", {"mode":"bad"}))


def test_remote_refs_not_allowed(tools):
    invalid=[{"type":"function","function":{"name":"f","parameters":{"type":"object","$ref":"https://example.com/schema.json"}}}]
    with pytest.raises(InvalidRequest): reg(invalid)


def test_message_context_preserves_tool_result(tools):
    prior={"role":"assistant","content":None,"tool_calls":[{"id":"call_abc","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"notes.txt\"}"}}]}
    messages=[{"role":"user","content":"read notes"},prior,{"role":"tool","tool_call_id":"call_abc","name":"read_file","content":"contents"}]
    prompt=make_prompt(messages,reg(tools))
    assert 'call_abc' in prompt and 'contents' in prompt and 'tool_call_id' in prompt
    assert 'tool_calls' in prompt and 'read_file' in prompt


def test_message_text_parts_supported(tools):
    prompt=make_prompt([{"role":"user","content":[{"type":"text","text":"hi"}]}],reg(tools))
    assert 'hi' in prompt


def test_image_input_rejected(tools):
    with pytest.raises(InvalidRequest):
        make_prompt([{"role":"user","content":[{"type":"image_url","image_url":{"url":"a"}}]}],reg(tools))


def test_tool_history_requires_identifier(tools):
    with pytest.raises(InvalidRequest):
        make_prompt([{"role":"tool","content":"foo"}],reg(tools))


def test_prompt_does_not_claim_tool_execution(tools):
    prompt=make_prompt([{"role":"user","content":"use tool"}],reg(tools))
    assert 'NOT executing tools' in prompt
    assert 'Hermes for permission and execution' in prompt


def test_reserved_name_function_can_be_forced():
    tools=[{"type":"function","function":{"name":"none","parameters":{"type":"object","properties":{}}}}]
    force={"type":"function","function":{"name":"none"}}
    assert reg(tools,force).enabled
    assert reg(tools,force).validate_response(call("none", {}))[1]
