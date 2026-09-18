"""Optimizer-only check on the public cases: reference directives go straight into solve(), bypassing the LLM."""
import json

import pytest

from app.optimizer import solve
from app.replay import replay
from app.schemas import DirectiveInterpretation, ScenarioRequest
from helpers import find_cases_file

CASES = json.loads(find_cases_file().read_text(encoding="utf-8"))["cases"]


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_reference_directives_reach_reference_cost(case):
    req = ScenarioRequest(**case["input"])
    ref = case["expected_output"]
    dirs = [DirectiveInterpretation(**d) for d in ref["directive_interpretation"]]

    plan, mode = solve(req.hours, req.battery, dirs)
    total_grid, total_cost, peak = replay(req.hours, req.battery, dirs, plan)  # raises on any violation

    assert mode == "optimal"
    # Cost is the scored quantity. total_grid/peak are NOT compared to the reference: many plans tie on cost
    # and differ in those (the judge only checks they are consistent with hourly_plan, which replay() computes).
    assert total_cost == pytest.approx(ref["total_cost_bdt"], abs=0.01)
    assert peak == max(p.grid_kwh for p in plan)
    assert total_grid == pytest.approx(sum(p.grid_kwh for p in plan))


def test_all_ten_cases_present():
    assert len(CASES) == 10
