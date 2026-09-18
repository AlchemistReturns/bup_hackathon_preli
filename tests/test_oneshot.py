import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
from openai import OpenAI
import pytest

from app import graph, llm_client, main
from app.guardrails import GuardrailError, validate_all
from app.interpretation_cache import InterpretationCache
from app.llm_protocol import expand_packed_directives
from test_optimizer import CASES


@pytest.mark.parametrize("row,kind,hours", [
    ([0, 0], "no_op", None),
    ([1, 0.25, 9, 12], "solar_reduction", [9, 10, 11]),
    ([2, 85.5, 18, 21], "minimum_battery_reserve", [18, 19, 20]),
    ([3, 0, 23, 2], "no_charge_window", [0, 1, 23]),
    ([4, 0, 0, 24], "no_discharge_window", list(range(24))),
    ([5, 31.25, 9, 11, 17, 20], "max_grid_window", [9, 10, 17, 18, 19]),
])
def test_numeric_protocol(row, kind, hours):
    result = validate_all(expand_packed_directives({"d": [row]}), 1, 200)[0]
    assert result.directive_type == kind
    if hours is None:
        assert result.structured_adjustment is None
    else:
        assert result.structured_adjustment["hours"] == hours


@pytest.mark.parametrize("row", [
    [], [0], [0, 0, 1], [6, 0], [-1, 0], [True, 0], [1.0, 0.2, 1, 2],
    ["solar", 0.2, 1, 2], [0, 1], [0, 0, 1, 2], [3, 1, 1, 2],
    [4, True, 1, 2], [2, "30", 1, 2], [2, float("nan"), 1, 2],
    [2, float("inf"), 1, 2], [1, 1.2, 1, 2], [5, -1, 1, 2],
    [2, 201, 1, 2], [3, 0], [3, 0, True, 2], [3, 0, 1.0, 2],
    [3, 0, 0, 25], [3, 0, 5, 5], [3, 0, 0, 1] * 15,
])
def test_invalid_numeric_protocol_rejected(row):
    with pytest.raises(GuardrailError):
        validate_all(expand_packed_directives({"d": [row]}), 1, 200)


def test_chat_sdk_request_and_decode(monkeypatch):
    monkeypatch.setattr(llm_client, "OPENAI_API_MODE", "chat")
    captured = []
    def handler(request):
        assert request.url.path == "/v1/chat/completions"
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "chat_test", "object": "chat.completion", "created": 0,
            "model": "gpt-4o-mini", "choices": [{"index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": '{"d":[[2,100,18,21]]}', "refusal": None}}],
        })
    with OpenAI(api_key="test-placeholder", max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
        monkeypatch.setattr(llm_client, "get_client", lambda: client)
        result = llm_client.interpret_notes(["Hold 50% from 6 to 9 PM"], 200)
    assert result[0]["structured_adjustment"] == {"minimum_energy_kwh": 100, "hours": [18, 19, 20]}
    request = captured[0]
    assert request["model"] == llm_client.OPENAI_MODEL
    assert request["response_format"]["json_schema"]["strict"] is True
    assert request["store"] is False
    assert json.loads(request["messages"][-1]["content"])["battery"]["capacity_kwh"] == 200


@pytest.mark.parametrize("finish,content,refusal", [
    ("length", '{"d":[[0,0]]}', None),
    ("content_filter", None, None),
    ("stop", None, "refused"),
    ("stop", "", None),
    ("stop", '{"d":[[2,NaN,1,2]]}', None),
    ("stop", "not JSON", None),
])
def test_chat_rejects_partial_refused_or_bad_output(monkeypatch, finish, content, refusal):
    monkeypatch.setattr(llm_client, "OPENAI_API_MODE", "chat")
    create = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish, message=SimpleNamespace(content=content, refusal=refusal))]))
    monkeypatch.setattr(llm_client, "get_client", lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))))
    with pytest.raises((ValueError, GuardrailError)):
        llm_client.interpret_notes(["Test"], 200)


def test_no_cache_means_every_request_interpreted(monkeypatch):
    monkeypatch.setattr(graph, "_cache", InterpretationCache(maxsize=0))
    call = Mock(return_value=CASES[0]["expected_output"]["directive_interpretation"])
    monkeypatch.setattr(graph, "interpret_notes", call)
    from fastapi.testclient import TestClient
    with TestClient(main.app) as client:
        for _ in range(2):
            result = client.post("/optimize-energy", json=CASES[0]["input"])
            assert result.status_code == 200
            assert result.headers["x-interpretation-cache"] == "disabled"
    assert call.call_count == 2


def test_startup_initializes_local_resources_without_paid_call(monkeypatch):
    client = Mock()
    initialize = Mock(return_value=client)
    compile_graph = Mock()
    monkeypatch.setattr(main, "OPENAI_API_KEY", "test-placeholder")
    monkeypatch.setattr(main, "get_client", initialize)
    monkeypatch.setattr(main, "get_graph", compile_graph)
    from fastapi.testclient import TestClient
    with TestClient(main.app) as http:
        assert http.get("/health").status_code == 200
    initialize.assert_called_once()
    compile_graph.assert_called_once()
    client.chat.completions.create.assert_not_called()
    client.responses.create.assert_not_called()
