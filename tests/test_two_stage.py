"""Two-stage objective: minimise cost (stage 1), then among cost-optimal plans minimise peak grid draw (stage 2).
Cost is primary and must never be put at risk by stage 2."""
import json
import random
import time

import pytest

from app.optimizer import compile_bounds, plan_cost, solve
from app.replay import replay
from app.schemas import DirectiveInterpretation, ScenarioRequest
from helpers import check_physics, find_cases_file
from test_fuzz import PHYSICS_TOL, _scenario

CASES = json.loads(find_cases_file().read_text(encoding="utf-8"))["cases"]
RESULTS = []


def peak(plan):
    return max(p.grid_kwh for p in plan)


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_public_cases_stage2_cost_equals_stage1_cost(case):
    req = ScenarioRequest(**case["input"])
    ref = case["expected_output"]
    dirs = [DirectiveInterpretation(**d) for d in ref["directive_interpretation"]]

    p1, m1 = solve(req.hours, req.battery, dirs, minimize_peak=False)
    p2, m2 = solve(req.hours, req.battery, dirs)
    c1, c2 = plan_cost(req.hours, p1), plan_cost(req.hours, p2)
    RESULTS.append((case["id"], ref["total_cost_bdt"], c1, c2, peak(p1), peak(p2), ref["peak_grid_kwh"]))

    assert m1 == m2 == "optimal"
    assert abs(c2 - c1) <= 1e-6, f"stage-2 cost drifted by {c2 - c1}"  # exact, not merely within 0.01
    assert c2 == pytest.approx(ref["total_cost_bdt"], abs=0.01)
    assert peak(p2) <= peak(p1) + 1e-9  # stage 2 can only lower the peak
    replay(req.hours, req.battery, dirs, p2)  # full directive-aware replay still passes
    assert peak(p2) == pytest.approx(ref["peak_grid_kwh"], abs=0.01)  # observed: matches the reference peaks too


def test_zz_print_stage_table():
    if not RESULTS:
        pytest.skip("run together with the parametrized cases")
    print(f"\n{'case':<10}{'ref cost':>11}{'stage1':>15}{'stage2':>15}{'delta':>11}{'pk1':>8}{'pk2':>8}{'ref pk':>8}")
    for cid, rc, c1, c2, p1, p2, rp in sorted(RESULTS):
        print(f"{cid:<10}{rc:>11.2f}{c1:>15.6f}{c2:>15.6f}{c2 - c1:>11.2e}{p1:>8.2f}{p2:>8.2f}{rp:>8.2f}")


def test_fuzz_stage2_never_costs_more_never_raises_peak_and_stays_physical():
    rng = random.Random(4242)
    n, lowered, kept_stage1 = 300, 0, 0
    worst = 0.0
    for i in range(n):
        hours, battery, dirs = _scenario(rng)
        p1, m1 = solve(hours, battery, dirs, minimize_peak=False)
        p2, m2 = solve(hours, battery, dirs)
        assert m1 == m2
        if m1 != "optimal":
            assert [x.model_dump() for x in p1] == [x.model_dump() for x in p2]  # relaxed mode has no stage 2
            continue
        c1, c2 = plan_cost(hours, p1), plan_cost(hours, p2)
        worst = max(worst, c2 - c1)
        assert c2 <= c1 + 1e-3, f"scenario {i}: stage-2 cost {c2} > stage-1 {c1}"
        assert peak(p2) <= peak(p1), f"scenario {i}: stage 2 raised the peak"
        assert check_physics(hours, battery, compile_bounds(hours, battery, dirs).eff_solar, p2, tol=PHYSICS_TOL) == []
        replay(hours, battery, dirs, p2)
        if peak(p2) < peak(p1) - 1e-6:
            lowered += 1
        if [x.model_dump() for x in p2] == [x.model_dump() for x in p1]:
            kept_stage1 += 1
    print(f"\nstage-2 fuzz: {n} scenarios, peak strictly lowered in {lowered}, worst cost delta {worst:.2e}")


def test_stage2_latency_is_small():
    case = CASES[1]
    req = ScenarioRequest(**case["input"])
    dirs = [DirectiveInterpretation(**d) for d in case["expected_output"]["directive_interpretation"]]
    t = time.perf_counter()
    for _ in range(5):
        solve(req.hours, req.battery, dirs)
    per = (time.perf_counter() - t) / 5
    print(f"\nsolve() with two stages: {per * 1000:.0f} ms")
    assert per < 2.0
