import asyncio
import copy
import json
import time
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import OpenAI
from pydantic import ValidationError

from app import graph, llm_client, main
from app.graph import InterpretationError, run_interpretation
from app.guardrails import GuardrailError, validate_all
from app.replay import ReplayError, replay_response
from app.schemas import HourEntry, OptimizeEnergyResponse, ScenarioRequest
from test_optimizer import CASES, prepare


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_public_api_pipeline(case, monkeypatch):
    notes = copy.deepcopy(case["expected_output"]["directive_interpretation"])
    call = Mock(return_value=notes)
    monkeypatch.setattr(graph, "interpret_notes", call)
    with TestClient(main.app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        response = client.post("/optimize-energy", json=case["input"])
    assert response.status_code == 200, response.text
    result = OptimizeEnergyResponse.model_validate(response.json())
    replay_response(ScenarioRequest.model_validate(case["input"]), result)
    assert result.total_cost_bdt == pytest.approx(case["expected_output"]["total_cost_bdt"], abs=0.01)
    assert call.call_args.kwargs["battery_capacity"] == case["input"]["battery"]["capacity_kwh"]


@pytest.mark.parametrize("kind,key", [
    ("minimum_battery_reserve", "minimum_energy_kwh"),
    ("max_grid_window", "max_grid_kwh"),
    ("solar_reduction", "factor"),
])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "10", -1, 10**400])
def test_bad_directive_numbers(kind, key, value):
    raw = [dict(note_index=0, applies=True, directive_type=kind,
                structured_adjustment={"hours": [1], key: value}, explanation="Test")]
    with pytest.raises(GuardrailError):
        validate_all(raw, 1, 200)


@pytest.mark.parametrize("field,value", [
    ("note_index", False), ("note_index", 0.0), ("directive_type", []),
    ("directive_type", {}), ("explanation", {}), ("explanation", ""),
])
def test_malformed_interpretation_is_controlled(field, value):
    entry = dict(note_index=0, applies=False, directive_type="no_op",
                 structured_adjustment=None, explanation="Irrelevant")
    entry[field] = value
    with pytest.raises(GuardrailError):
        validate_all([entry], 1, 200)


@pytest.mark.parametrize("field", ["demand_kwh", "solar_kwh", "tariff_bdt_per_kwh"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_request_rejected(field, value):
    entry = dict(hour=0, demand_kwh=1, solar_kwh=0, tariff_bdt_per_kwh=1)
    entry[field] = value
    with pytest.raises(ValidationError):
        HourEntry(**entry)


def test_failed_interpretation_never_reaches_optimizer(monkeypatch, caplog):
    call = Mock(side_effect=RuntimeError("SECRET-provider-details"))
    solve = Mock()
    monkeypatch.setattr(graph, "interpret_notes", call)
    monkeypatch.setattr(main, "solve", solve)
    with TestClient(main.app) as client:
        response = client.post("/optimize-energy", json=CASES[0]["input"])
    assert response.status_code == 500
    assert call.call_count == graph.LLM_MAX_RETRIES + 1
    solve.assert_not_called()
    assert "SECRET" not in response.text
    assert "SECRET" not in caplog.text
    assert "hourly_plan" not in response.json()


def test_invalid_then_valid_interpretation_retries(monkeypatch):
    request, directives = prepare(CASES[0])
    call = Mock(side_effect=[[], [d.model_dump() for d in directives]])
    monkeypatch.setattr(graph, "interpret_notes", call)
    result = run_interpretation(request.operator_notes, request.battery.capacity_kwh)
    assert result == directives
    assert call.call_count == 2
    assert "expected exactly" in call.call_args.kwargs["feedback"]


def test_deadline_prevents_model_call(monkeypatch):
    call = Mock()
    monkeypatch.setattr(graph, "interpret_notes", call)
    monkeypatch.setattr(graph, "LLM_BUDGET_SECONDS", 0)
    with pytest.raises(InterpretationError):
        run_interpretation(["Do not charge at noon"], 200)
    call.assert_not_called()


def test_sdk_request_contains_capacity_and_strict_schema(monkeypatch):
    monkeypatch.setattr(llm_client, "OPENAI_API_MODE", "responses")
    captured = []
    model_directives = [[2, 100, 18, 21]]
    def handler(request):
        captured.append(json.loads(request.content))
        return httpx.Response(200, json={
            "id": "resp_test", "object": "response", "created_at": 0,
            "model": "gpt-4o-mini", "status": "completed",
            "output": [{
                "id": "msg_test", "type": "message", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "annotations": [],
                             "text": json.dumps({"d": model_directives})}],
            }],
        })
    with OpenAI(api_key="test-placeholder", max_retries=0,
                http_client=httpx.Client(transport=httpx.MockTransport(handler))) as client:
        monkeypatch.setattr(llm_client, "get_client", lambda: client)
        result = llm_client.interpret_notes(CASES[2]["input"]["operator_notes"], 200)
    assert result[0]["structured_adjustment"]["minimum_energy_kwh"] == 100
    assert result[0]["structured_adjustment"]["hours"] == [18, 19, 20]
    prompt = json.loads(captured[0]["input"][-1]["content"])
    assert prompt["battery"]["capacity_kwh"] == 200
    assert captured[0]["text"]["format"]["strict"] is True
    assert captured[0]["store"] is False


@pytest.mark.parametrize("status,text", [
    ("incomplete", '{"directives": []}'),
    ("completed", "refused"),
    ("completed", '{"directives": NaN}'),
    ("completed", '{"directives": null}'),
])
def test_bad_model_response_rejected(monkeypatch, status, text):
    monkeypatch.setattr(llm_client, "OPENAI_API_MODE", "responses")
    create = Mock(return_value=SimpleNamespace(status=status, output_text=text))
    monkeypatch.setattr(llm_client, "get_client", lambda: SimpleNamespace(responses=SimpleNamespace(create=create)))
    with pytest.raises((ValueError, GuardrailError)):
        llm_client.interpret_notes(["Test"], 200)


@pytest.mark.parametrize("body", ["{broken", "null", '{"hours": []}', '{"value": NaN}'])
def test_bad_http_input_returns_controlled_400(body):
    with TestClient(main.app) as client:
        response = client.post("/optimize-energy", content=body, headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    assert response.json() == {"detail": "structurally invalid request"}


def test_independent_response_replay_checks_totals():
    request, directives = prepare(CASES[0])
    response = OptimizeEnergyResponse.model_validate(CASES[0]["expected_output"])
    response.total_cost_bdt += 10
    with pytest.raises(ReplayError):
        replay_response(request, response)


def test_health_remains_responsive_and_request_deadline(monkeypatch):
    monkeypatch.setattr(main, "REQUEST_TIMEOUT_SECONDS", 0.03)
    def slow_work(payload):
        time.sleep(0.15)
        raise InterpretationError("delayed test")
    monkeypatch.setattr(main, "_optimize", slow_work)
    async def exercise():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app), base_url="http://test") as client:
            pending = asyncio.create_task(client.post("/optimize-energy", json=CASES[0]["input"]))
            await asyncio.sleep(0.01)
            started = time.perf_counter()
            health = await client.get("/health")
            assert time.perf_counter() - started < 0.1
            assert health.status_code == 200
            result = await pending
            assert result.status_code == 500
            assert result.json()["detail"] == "request deadline exceeded"
    asyncio.run(exercise())
