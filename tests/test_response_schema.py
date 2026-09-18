"""Section 10 / 11.3: exact field-level compliance of the HTTP response, checked on the RAW response text as well as
the parsed object (so 0 vs 0.0 and -0.0 are visible), across optimal, relaxed, safe-failure and fallback plans."""
import copy
import json
import math
import random
import re

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.schemas import DirectiveInterpretation
from helpers import directive, find_cases_file
from test_fuzz import _scenario

CASES = json.loads(find_cases_file().read_text(encoding="utf-8"))["cases"]
client = TestClient(main.app, raise_server_exceptions=False)

TOP = {"scenario_id", "directive_interpretation", "hourly_plan", "total_grid_kwh", "total_cost_bdt",
       "peak_grid_kwh", "plan_summary"}
DIR = {"note_index", "applies", "directive_type", "structured_adjustment", "explanation"}
ROW = {"hour", "grid_kwh", "solar_used_kwh", "battery_action", "battery_kwh", "battery_energy_after_kwh"}
FLOATS = ("grid_kwh", "solar_used_kwh", "battery_kwh", "battery_energy_after_kwh")
TYPES = {"solar_reduction", "minimum_battery_reserve", "no_charge_window", "no_discharge_window",
         "max_grid_window", "no_op"}


def stub(monkeypatch, directives):
    monkeypatch.setattr(main, "run_interpretation", lambda operator_notes, battery_capacity: directives)


def no_ops(n):
    return [DirectiveInterpretation(note_index=i, applies=False, directive_type="no_op") for i in range(n)]


def check_response(resp, request_json):
    assert resp.status_code == 200, resp.text[:300]
    text, body = resp.text, resp.json()

    assert set(body) == TOP, f"missing/extra top-level fields: {set(body) ^ TOP}"
    assert body["scenario_id"] == request_json["scenario_id"]
    assert isinstance(body["plan_summary"], str) and body["plan_summary"].strip()

    # ---- interpretation: one entry per note, ordered, every field present (structured_adjustment present even if null)
    di = body["directive_interpretation"]
    assert len(di) == len(request_json["operator_notes"])
    for i, d in enumerate(di):
        assert set(d) == DIR and d["note_index"] == i and d["directive_type"] in TYPES
        assert isinstance(d["applies"], bool) and isinstance(d["explanation"], str)
        if d["directive_type"] == "no_op":
            assert d["applies"] is False and d["structured_adjustment"] is None
        else:
            assert d["applies"] is True and isinstance(d["structured_adjustment"], dict)

    # ---- plan: 24 unique hours 0..23, exact keys, float types, finite, non-negative
    plan = body["hourly_plan"]
    assert [p["hour"] for p in plan] == list(range(24))
    for p in plan:
        assert set(p) == ROW
        assert p["battery_action"] in ("charge", "discharge", "idle")
        for f in FLOATS:
            assert type(p[f]) is float, f"hour {p['hour']} {f} serialised as {type(p[f]).__name__}"
            assert math.isfinite(p[f]) and p[f] >= 0
        if p["battery_action"] == "idle":
            assert p["battery_kwh"] == 0.0

    # ---- raw-text checks: no -0.0, no NaN/Infinity tokens, idle battery_kwh is literally 0.0
    assert "-0.0" not in text and "NaN" not in text and "Infinity" not in text
    for m in re.finditer(r'"battery_action":"idle","battery_kwh":([^,}]+)', text):
        assert m.group(1) == "0.0", f"idle battery_kwh serialised as {m.group(1)!r}"
    for f in ("total_grid_kwh", "total_cost_bdt", "peak_grid_kwh"):
        assert type(body[f]) is float and math.isfinite(body[f]) and body[f] >= 0

    # ---- Section 11.3: totals recompute from hourly_plan within the 0.01 tolerance
    tariff = {h["hour"]: h["tariff_bdt_per_kwh"] for h in request_json["hours"]}
    assert body["total_grid_kwh"] == pytest.approx(sum(p["grid_kwh"] for p in plan), abs=0.01)
    assert body["total_cost_bdt"] == pytest.approx(sum(p["grid_kwh"] * tariff[p["hour"]] for p in plan), abs=0.01)
    assert body["peak_grid_kwh"] == pytest.approx(max(p["grid_kwh"] for p in plan), abs=0.01)
    return body


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_public_case_responses_are_field_exact(monkeypatch, case):
    stub(monkeypatch, [DirectiveInterpretation(**d) for d in case["expected_output"]["directive_interpretation"]])
    check_response(client.post("/optimize-energy", json=case["input"]), case["input"])


def test_relaxed_plan_response_is_field_exact_with_nonempty_summary(monkeypatch):
    req = copy.deepcopy(CASES[0]["input"])
    n = len(req["operator_notes"])
    stub(monkeypatch, [directive("max_grid_window", hours=list(range(24)), max_grid_kwh=1)] + no_ops(n)[1:])
    body = check_response(client.post("/optimize-energy", json=req), req)
    assert "minimum-violation" in body["plan_summary"]


def test_fallback_plan_response_is_field_exact_with_nonempty_summary(monkeypatch):
    req = copy.deepcopy(CASES[0]["input"])
    stub(monkeypatch, no_ops(len(req["operator_notes"])))
    real = main.solve

    def broken(hours, battery, directives):
        plan, mode = real(hours, battery, directives)
        plan[3] = plan[3].model_copy(update={"grid_kwh": plan[3].grid_kwh + 99})
        return plan, mode

    monkeypatch.setattr(main, "solve", broken)
    body = check_response(client.post("/optimize-energy", json=req), req)
    assert "fallback" in body["plan_summary"]


def test_zero_demand_zero_solar_scenario_has_no_negative_zero_anywhere(monkeypatch):
    req = copy.deepcopy(CASES[0]["input"])
    for h in req["hours"]:
        h.update(demand_kwh=0, solar_kwh=0, tariff_bdt_per_kwh=0)
    stub(monkeypatch, no_ops(len(req["operator_notes"])))
    check_response(client.post("/optimize-energy", json=req), req)


def test_negative_zero_in_request_does_not_leak_into_response(monkeypatch):
    text = json.dumps(CASES[0]["input"]).replace('"solar_kwh": 0,', '"solar_kwh": -0.0,')
    assert '-0.0' in text
    stub(monkeypatch, no_ops(len(CASES[0]["input"]["operator_notes"])))
    r = client.post("/optimize-energy", content=text, headers={"content-type": "application/json"})
    assert r.status_code == 200 and "-0.0" not in r.text


def test_random_scenarios_all_field_exact(monkeypatch):
    """150 random scenarios (random directives, ~15% forcing the relaxed path) through the real HTTP handler."""
    rng = random.Random(777)
    relaxed = 0
    base = CASES[0]["input"]
    for i in range(150):
        hours, battery, dirs = _scenario(rng)
        n = max(1, len(dirs))
        entries = dirs + no_ops(n)[len(dirs):]
        for j, d in enumerate(entries):
            entries[j] = d.model_copy(update={"note_index": j})
        req = {
            "scenario_id": f"FUZZ-{i}", "operator_notes": [f"note {k}" for k in range(n)],
            "hours": [h.model_dump() for h in hours], "battery": battery.model_dump(),
        }
        monkeypatch.setattr(main, "run_interpretation", lambda operator_notes, battery_capacity, e=entries: e)
        body = check_response(client.post("/optimize-energy", json=req), req)
        relaxed += "minimum-violation" in body["plan_summary"]
    print(f"\nresponse fuzz: 150 scenarios field-exact; {relaxed} were relaxed-mode responses")
    assert relaxed > 0
