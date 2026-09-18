import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError

import pytest

from app import graph, llm_client
from app.guardrails import GuardrailError
from app.optimizer import solve
from app.replay import replay
from app.schemas import OptimizeEnergyResponse
from app.time_windows import normalize_time_windows
from scripts import check_samples
from test_optimizer import CASES, prepare


def raw(windows, kind="no_charge_window", **values):
    return [dict(
        note_index=0, applies=True, directive_type=kind,
        structured_adjustment={"time_windows": windows, **values},
        explanation="Interpreted the note",
    )]


@pytest.mark.parametrize("start,end,expected", [
    (12, 14, [12, 13]),        # SAMPLE-01 regression
    (14, 16, [14, 15]),        # SAMPLE-06 regression
    (18, 22, [18, 19, 20, 21]), # SAMPLE-07 regression
    (18, 20, [18, 19]),        # First run's SAMPLE-04 regression
    (0, 1, [0]), (23, 24, [23]), (0, 24, list(range(24))),
    (23, 2, [0, 1, 23]), (11, 12, [11]), (12, 13, [12]),
])
def test_half_open_window_expansion(start, end, expected):
    result = normalize_time_windows(raw([{"start_hour": start, "end_hour": end}]))
    assert result[0]["structured_adjustment"] == {"hours": expected}


def test_disjoint_overlapping_windows_are_unique_and_sorted():
    result = normalize_time_windows(raw([
        {"start_hour": 18, "end_hour": 20},
        {"start_hour": 3, "end_hour": 5},
        {"start_hour": 19, "end_hour": 21},
    ]))
    assert result[0]["structured_adjustment"]["hours"] == [3, 4, 18, 19, 20]


@pytest.mark.parametrize("window", [
    {"start_hour": 5, "end_hour": 5},
    {"start_hour": -1, "end_hour": 4},
    {"start_hour": 24, "end_hour": 25},
    {"start_hour": 23, "end_hour": 0},
    {"start_hour": True, "end_hour": 4},
    {"start_hour": 1.0, "end_hour": 4},
    {"start_hour": "1", "end_hour": 4},
    {"start_hour": 1, "end_hour": float("nan")},
    {"start_hour": 1}, {"hours": [1, 2]}, None,
])
def test_invalid_windows_are_rejected(window):
    with pytest.raises(GuardrailError):
        normalize_time_windows(raw([window]))


def test_old_hours_output_is_not_silently_accepted():
    entry = raw([{"start_hour": 12, "end_hour": 14}])
    entry[0]["structured_adjustment"] = {"hours": [12, 13, 14]}
    with pytest.raises(GuardrailError):
        normalize_time_windows(entry)


def test_normalizer_preserves_value_and_input():
    original = raw([{"start_hour": 9, "end_hour": 11}], "solar_reduction", factor=0.25)
    before = copy.deepcopy(original)
    result = normalize_time_windows(original)
    assert result[0]["structured_adjustment"] == {"hours": [9, 10], "factor": 0.25}
    assert original == before


def test_no_op_remains_null():
    entry = dict(note_index=0, applies=False, directive_type="no_op",
                 structured_adjustment=None, explanation="Unrelated note")
    assert normalize_time_windows([entry]) == [entry]


@pytest.mark.parametrize("start,end", [(12, 14), (14, 16), (18, 22)])
def test_model_output_goes_through_actual_normalizer(monkeypatch, start, end):
    response = SimpleNamespace(status="completed", output_text=json.dumps({
        "d": [{"t": "no_charge", "w": [{"s": start, "e": end}], "v": None}],
    }))
    client = SimpleNamespace(responses=SimpleNamespace(create=Mock(return_value=response)))
    monkeypatch.setattr(llm_client, "get_client", lambda: client)
    entries = llm_client.interpret_notes(["Synthetic note"], 200)
    assert entries[0]["structured_adjustment"]["hours"] == list(range(start, end))


def test_bad_boundary_feedback_reaches_retry(monkeypatch):
    good = [dict(note_index=0, applies=True, directive_type="no_charge_window",
                 structured_adjustment={"hours": [9, 10]}, explanation="Maintenance")]
    call = Mock(side_effect=[GuardrailError("time boundaries must be integers"), good])
    monkeypatch.setattr(graph, "interpret_notes", call)
    result = graph.run_interpretation(["Charging unavailable 9 AM until 11 AM"], 200)
    assert result[0].structured_adjustment["hours"] == [9, 10]
    assert call.call_args.kwargs["feedback"] == "time boundaries must be integers"


def test_checker_does_not_confuse_self_consistency_with_ground_truth():
    case = CASES[0]
    request, truth = prepare(case)
    # Ignore the solar restriction: a self-consistent plan under wrong directives.
    wrong = copy.deepcopy(truth)
    wrong[0].structured_adjustment["factor"] = 1.0
    plan = solve(request.hours, request.battery, wrong)
    grid, cost, peak = replay(request.hours, request.battery, wrong, plan)
    result = OptimizeEnergyResponse(
        scenario_id=request.scenario_id, directive_interpretation=wrong,
        hourly_plan=plan, total_grid_kwh=grid, total_cost_bdt=cost,
        peak_grid_kwh=peak, plan_summary="Synthetic bad interpretation",
    )
    evaluated = check_samples.evaluate_response(case, result)
    assert evaluated["reported_plan_valid"]
    assert not evaluated["interpretation"]
    assert not evaluated["ground_truth_valid"]
    assert not evaluated["passed"]


def test_runner_continues_after_http_failure(monkeypatch):
    second = json.dumps(CASES[1]["expected_output"]).encode()
    class Response:
        status = 200
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self):
            return second
    call = Mock(side_effect=[HTTPError("http://test", 500, "error", {}, None), Response()])
    monkeypatch.setattr(check_samples, "urlopen", call)
    records = check_samples.run_cases(CASES[:2], "http://test")
    assert len(records) == 2
    assert records[0]["http_status"] == 500
    assert not records[0]["passed"]
    assert records[1]["passed"]
