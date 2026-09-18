"""Section 07 / 6.1: every malformed or structurally invalid request must return exactly 400 (never 422, never 500),
must not reach interpretation/optimization, and must not leak stack traces. Valid boundary requests must return 200."""
import copy
import json
import random

import pytest
from fastapi.testclient import TestClient

import app.main as main
from app.schemas import DirectiveInterpretation
from helpers import find_cases_file

BASE = json.loads(find_cases_file().read_text(encoding="utf-8"))["cases"][0]["input"]
client = TestClient(main.app, raise_server_exceptions=False)


def no_op_all(operator_notes, battery_capacity):
    return [DirectiveInterpretation(note_index=i, applies=False, directive_type="no_op", explanation="stub")
            for i in range(len(operator_notes))]


@pytest.fixture(autouse=True)
def stub(monkeypatch):
    monkeypatch.setattr(main, "run_interpretation", no_op_all)


@pytest.fixture
def must_not_interpret(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("invalid request reached the interpretation step")
    monkeypatch.setattr(main, "run_interpretation", boom)


def req(mutate=None):
    r = copy.deepcopy(BASE)
    if mutate:
        mutate(r)
    return r


def post(body):
    return client.post("/optimize-energy", json=body)


def post_raw(text, ctype="application/json"):
    return client.post("/optimize-energy", content=text, headers={"content-type": ctype})


def setp(path, value):
    def m(r):
        cur = r
        for k in path[:-1]:
            cur = cur[k]
        cur[path[-1]] = value
    return m


def delp(path):
    def m(r):
        cur = r
        for k in path[:-1]:
            cur = cur[k]
        del cur[path[-1]]
    return m


def dup_hour(r):
    r["hours"][7]["hour"] = 6  # two entries for hour 6, none for hour 7


def gap_hour(r):
    r["hours"][7]["hour"] = 24  # hour 7 missing, 24 present


def drop_last(r):
    r["hours"].pop()


def add_one(r):
    r["hours"].append({**r["hours"][0], "hour": 24})


def add_dup_zero(r):
    r["hours"].append(copy.deepcopy(r["hours"][0]))


INVALID = {
    # ---- hours array shape
    "23-hours": drop_last, "25-hours-hour24": add_one, "25-hours-dup0": add_dup_zero,
    "0-hours": setp(["hours"], []), "hours-null": setp(["hours"], None), "hours-not-list": setp(["hours"], "abc"),
    "hours-object": setp(["hours"], {"0": 1}), "duplicate-hour": dup_hour, "missing-hour-7": gap_hour,
    "hour-negative": setp(["hours", 3, "hour"], -1), "hour-24": setp(["hours", 3, "hour"], 24),
    "hour-string": setp(["hours", 3, "hour"], "three"), "hour-null": setp(["hours", 3, "hour"], None),
    "hour-key-missing": delp(["hours", 3, "hour"]),
    "hour-fractional": setp(["hours", 3, "hour"], 3.5),
    "hours-contains-null": setp(["hours", 3], None),
    # ---- notes
    "0-notes": setp(["operator_notes"], []), "4-notes": setp(["operator_notes"], ["a", "b", "c", "d"]),
    "empty-string-note": setp(["operator_notes"], [""]), "blank-note": setp(["operator_notes"], ["   \t\n"]),
    "one-empty-of-two": setp(["operator_notes"], ["ok", ""]),
    "note-not-string": setp(["operator_notes"], [5]), "note-null": setp(["operator_notes"], [None]),
    "notes-is-string": setp(["operator_notes"], "do not charge"), "notes-null": setp(["operator_notes"], None),
    "notes-missing": delp(["operator_notes"]),
    # ---- negative numbers
    "demand-negative": setp(["hours", 5, "demand_kwh"], -1), "solar-negative": setp(["hours", 5, "solar_kwh"], -0.5),
    "tariff-negative": setp(["hours", 5, "tariff_bdt_per_kwh"], -3),
    "demand-string": setp(["hours", 5, "demand_kwh"], "lots"), "demand-null": setp(["hours", 5, "demand_kwh"], None),
    "solar-missing": delp(["hours", 5, "solar_kwh"]), "tariff-missing": delp(["hours", 5, "tariff_bdt_per_kwh"]),
    "demand-missing": delp(["hours", 5, "demand_kwh"]),
    "charge-negative": setp(["battery", "max_charge_kwh_per_hour"], -1),
    "discharge-negative": setp(["battery", "max_discharge_kwh_per_hour"], -1),
    "initial-negative": setp(["battery", "initial_energy_kwh"], -1),
    "min-negative": setp(["battery", "minimum_energy_kwh"], -1),
    "capacity-zero": setp(["battery", "capacity_kwh"], 0), "capacity-negative": setp(["battery", "capacity_kwh"], -100),
    # ---- battery consistency
    "min-greater-than-capacity": lambda r: r["battery"].update(minimum_energy_kwh=r["battery"]["capacity_kwh"] + 1),
    "initial-above-capacity": lambda r: r["battery"].update(initial_energy_kwh=r["battery"]["capacity_kwh"] + 0.5),
    "initial-below-minimum": lambda r: r["battery"].update(
        minimum_energy_kwh=50, initial_energy_kwh=49.9, capacity_kwh=200),
    "battery-missing": delp(["battery"]), "battery-null": setp(["battery"], None), "battery-list": setp(["battery"], []),
    "battery-field-missing": delp(["battery", "capacity_kwh"]),
    "battery-field-missing-2": delp(["battery", "max_discharge_kwh_per_hour"]),
    "battery-field-string": setp(["battery", "capacity_kwh"], "big"),
    # ---- top-level
    "scenario-id-missing": delp(["scenario_id"]), "scenario-id-int": setp(["scenario_id"], 7),
    "scenario-id-null": setp(["scenario_id"], None),
}


@pytest.mark.parametrize("name", INVALID)
def test_invalid_request_is_400(name, must_not_interpret):
    r = post(req(INVALID[name]))
    assert r.status_code == 400, f"{name}: got {r.status_code} {r.text[:200]}"
    assert "Traceback" not in r.text and "File \"" not in r.text


@pytest.mark.parametrize("text", [
    "", "   ", "{not json", '{"scenario_id": "x",', "null", "[]", "[1,2,3]", '"string"', "42", "true",
    "<xml/>", "{'single': 'quotes'}",
], ids=repr)
def test_malformed_or_non_object_json_is_400(text, must_not_interpret):
    r = post_raw(text)
    assert r.status_code == 400, f"got {r.status_code} {r.text[:200]}"


@pytest.mark.parametrize("ctype", ["text/plain", "application/x-www-form-urlencoded"])
def test_wrong_content_type_is_not_a_500(ctype, must_not_interpret):
    r = post_raw(json.dumps(BASE), ctype)
    assert r.status_code in (400, 415, 422), r.status_code  # never 500; spec has no wording for content type


def test_missing_content_type_with_valid_json_is_leniently_accepted():
    # FastAPI parses a header-less body as JSON; a judge that omits the header still gets a real answer.
    assert post_raw(json.dumps(BASE), "").status_code == 200


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
@pytest.mark.parametrize("field", ["demand_kwh", "solar_kwh", "tariff_bdt_per_kwh"])
def test_non_finite_hour_values_are_400(token, field, must_not_interpret):
    text = json.dumps(BASE)
    first = BASE["hours"][0][field]
    needle = f'"{field}": {json.dumps(first)}'
    assert needle in text
    r = post_raw(text.replace(needle, f'"{field}": {token}', 1))
    assert r.status_code == 400, f"{field}={token}: got {r.status_code}"


@pytest.mark.parametrize("token", ["NaN", "Infinity"])
@pytest.mark.parametrize("field", ["capacity_kwh", "initial_energy_kwh", "minimum_energy_kwh",
                                   "max_charge_kwh_per_hour", "max_discharge_kwh_per_hour"])
def test_non_finite_battery_values_are_400(token, field, must_not_interpret):
    text = json.dumps(BASE)
    needle = f'"{field}": {json.dumps(BASE["battery"][field])}'
    assert needle in text
    r = post_raw(text.replace(needle, f'"{field}": {token}', 1))
    assert r.status_code == 400, f"{field}={token}: got {r.status_code}"


# ---------------------------------------------------------------- valid boundary requests must still be 200
def _all_hours(r, **kw):
    for h in r["hours"]:
        h.update(kw)


VALID = {
    "one-note": setp(["operator_notes"], ["Only one note."]),
    "three-notes": setp(["operator_notes"], ["a", "b", "c"]),
    "initial-equals-capacity": lambda r: r["battery"].update(initial_energy_kwh=r["battery"]["capacity_kwh"]),
    "initial-equals-minimum": lambda r: r["battery"].update(initial_energy_kwh=r["battery"]["minimum_energy_kwh"]),
    "minimum-zero": lambda r: r["battery"].update(minimum_energy_kwh=0),
    "min-equals-init-equals-capacity": lambda r: r["battery"].update(
        minimum_energy_kwh=r["battery"]["capacity_kwh"], initial_energy_kwh=r["battery"]["capacity_kwh"]),
    "zero-charge-rate": setp(["battery", "max_charge_kwh_per_hour"], 0),
    "zero-discharge-rate": setp(["battery", "max_discharge_kwh_per_hour"], 0),
    "zero-both-rates": lambda r: r["battery"].update(max_charge_kwh_per_hour=0, max_discharge_kwh_per_hour=0),
    "all-zero-demand": lambda r: _all_hours(r, demand_kwh=0),
    "all-zero-tariff": lambda r: _all_hours(r, tariff_bdt_per_kwh=0),
    "all-zero-solar": lambda r: _all_hours(r, solar_kwh=0),
    "integer-valued": lambda r: _all_hours(r, demand_kwh=100, solar_kwh=0, tariff_bdt_per_kwh=5),
    "extra-unknown-fields-ignored": lambda r: r.update(unknown_field=1),
    "shuffled-hours": lambda r: random.Random(1).shuffle(r["hours"]),
    "reversed-hours": lambda r: r["hours"].reverse(),
    "unicode-note": setp(["operator_notes"], ["ব্যাটারি চার্জ করবেন না — 2 PM থেকে 4 PM"]),
    "very-long-note": setp(["operator_notes"], ["x " * 5000]),
    "huge-values": lambda r: _all_hours(r, demand_kwh=1e9, solar_kwh=1e9, tariff_bdt_per_kwh=1e3),
}


@pytest.mark.parametrize("name", VALID)
def test_valid_boundary_request_is_200_and_wellformed(name):
    r = post(req(VALID[name]))
    assert r.status_code == 200, f"{name}: {r.status_code} {r.text[:300]}"
    body = r.json()
    assert [p["hour"] for p in body["hourly_plan"]] == list(range(24))
    assert len(body["directive_interpretation"]) == len(req(VALID[name])["operator_notes"])
