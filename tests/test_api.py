"""HTTP-level tests with the LLM interpretation step stubbed (the live LLM path is NOT exercised here)."""
import json

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.replay import replay
from app.schemas import DirectiveInterpretation, ScenarioRequest
from helpers import directive, find_cases_file

CASES = json.loads(find_cases_file().read_text(encoding="utf-8"))["cases"]
client = TestClient(main.app)


def _stub(monkeypatch, directives):
    monkeypatch.setattr(main, "run_interpretation", lambda operator_notes, battery_capacity: directives)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok"}


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_optimize_with_reference_interpretation(monkeypatch, case):
    ref = case["expected_output"]
    _stub(monkeypatch, [DirectiveInterpretation(**d) for d in ref["directive_interpretation"]])
    r = client.post("/optimize-energy", json=case["input"])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["scenario_id"] == case["input"]["scenario_id"]
    assert len(body["hourly_plan"]) == 24
    assert body["total_cost_bdt"] == pytest.approx(ref["total_cost_bdt"], abs=0.01)
    assert "relaxed" not in body["plan_summary"] and "fallback" not in body["plan_summary"]


@pytest.mark.parametrize("body", [{"scenario_id": "x"}, {"scenario_id": "x", "operator_notes": [], "hours": []}])
def test_structurally_invalid_body_is_400(body):
    assert client.post("/optimize-energy", json=body).status_code == 400


def test_malformed_json_is_400():
    r = client.post("/optimize-energy", content="{not json", headers={"content-type": "application/json"})
    assert r.status_code == 400


def test_infeasible_directives_return_200_with_relaxed_note(monkeypatch):
    case = CASES[0]
    first = directive("max_grid_window", hours=list(range(24)), max_grid_kwh=1)
    rest = [DirectiveInterpretation(note_index=i, applies=False, directive_type="no_op")
            for i in range(1, len(case["input"]["operator_notes"]))]
    _stub(monkeypatch, [first] + rest)
    r = client.post("/optimize-energy", json=case["input"])
    assert r.status_code == 200, r.text
    assert "minimum-violation" in r.json()["plan_summary"]


def test_replay_failure_falls_back_to_grid_only_plan(monkeypatch):
    case = CASES[0]
    req = ScenarioRequest(**case["input"])
    _stub(monkeypatch, [DirectiveInterpretation(note_index=i, applies=False, directive_type="no_op")
                        for i in range(len(case["input"]["operator_notes"]))])
    real_solve = main.solve

    def bad_solve(hours, battery, directives):
        plan, mode = real_solve(hours, battery, directives)
        plan[5] = plan[5].model_copy(update={"grid_kwh": plan[5].grid_kwh + 50})  # break energy balance
        return plan, mode

    monkeypatch.setattr(main, "solve", bad_solve)
    r = client.post("/optimize-energy", json=case["input"])
    assert r.status_code == 200, r.text
    body = r.json()
    assert "fallback" in body["plan_summary"]
    assert all(p["battery_action"] == "idle" and p["solar_used_kwh"] == 0 for p in body["hourly_plan"])
    assert [p["grid_kwh"] for p in body["hourly_plan"]] == [h.demand_kwh for h in sorted(req.hours, key=lambda h: h.hour)]
    assert body["total_cost_bdt"] == pytest.approx(sum(h.demand_kwh * h.tariff_bdt_per_kwh for h in req.hours), abs=0.01)
