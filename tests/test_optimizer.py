import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.optimize import linprog

from app import optimizer
from app.guardrails import validate_all
from app.optimizer import OptimizationError, solve
from app.replay import ReplayError, replay
from app.schemas import HourlyPlanEntry, ScenarioRequest

CASES = json.loads((Path(__file__).parent / "fixtures/public_cases.json").read_text(encoding="utf-8"))["cases"]


def prepare(case):
    request = ScenarioRequest.model_validate(case["input"])
    directives = validate_all(
        copy.deepcopy(case["expected_output"]["directive_interpretation"]),
        len(request.operator_notes), request.battery.capacity_kwh,
    )
    return request, directives


@pytest.mark.parametrize("case", CASES, ids=lambda c: c["id"])
def test_public_optimal_cost(case):
    request, directives = prepare(case)
    plan = solve(request.hours, request.battery, directives)
    # Match physical validity and optimal cost, never the exact action sequence.
    serialized = [HourlyPlanEntry.model_validate_json(p.model_dump_json()) for p in plan]
    totals = replay(request.hours, request.battery, directives, serialized)
    assert totals[1] == pytest.approx(case["expected_output"]["total_cost_bdt"], abs=0.01)


def test_exactly_48_variables_and_shared_sparse_pattern(monkeypatch):
    original = optimizer.linprog
    calls = []
    def capture(costs, **kwargs):
        calls.append((len(costs), kwargs["A_ub"], kwargs["A_eq"]))
        return original(costs, **kwargs)
    monkeypatch.setattr(optimizer, "linprog", capture)
    for case in CASES[:2]:
        request, directives = prepare(case)
        solve(request.hours, request.battery, directives)
    assert [c[0] for c in calls] == [48, 48]
    assert calls[0][1] is calls[1][1]
    assert calls[0][2] is calls[1][2]


def reference_120_variable_lp(request, directives):
    """Independent oracle: explicit grid, solar, charge, discharge and energy.

    This intentionally does NOT import apply_directives or the reduced matrix.
    Comparing objectives avoids relying on a shared elimination/sign convention.
    """
    rows = sorted(request.hours, key=lambda row: row.hour)
    battery = request.battery
    bounds = (
        [(0, None) for _ in rows]
        + [(0, row.solar_kwh) for row in rows]
        + [(0, battery.max_charge_kwh_per_hour) for _ in rows]
        + [(0, battery.max_discharge_kwh_per_hour) for _ in rows]
        + [(battery.minimum_energy_kwh, battery.capacity_kwh) for _ in rows]
    )
    for directive in directives:
        if not directive.applies:
            continue
        adjustment = directive.structured_adjustment
        for h in adjustment["hours"]:
            kind = directive.directive_type
            if kind == "solar_reduction":
                bounds[24+h] = (0, bounds[24+h][1] * adjustment["factor"])
            elif kind == "no_charge_window":
                bounds[48+h] = (0, 0)
            elif kind == "no_discharge_window":
                bounds[72+h] = (0, 0)
            elif kind == "minimum_battery_reserve":
                bounds[96+h] = (max(bounds[96+h][0], adjustment["minimum_energy_kwh"]), battery.capacity_kwh)
            elif kind == "max_grid_window":
                previous = bounds[h][1]
                bounds[h] = (0, min(previous if previous is not None else np.inf, adjustment["max_grid_kwh"]))
    matrix = np.zeros((49, 120))
    rhs = np.zeros(49)
    for h, row in enumerate(rows):
        matrix[h, [h, 24+h, 48+h, 72+h]] = [1, 1, -1, 1]
        rhs[h] = row.demand_kwh
        matrix[24+h, [96+h, 48+h, 72+h]] = [1, -1, 1]
        if h:
            matrix[24+h, 96+h-1] = -1
        else:
            rhs[24+h] = battery.initial_energy_kwh
    matrix[48, 119] = 1
    rhs[48] = battery.initial_energy_kwh
    return linprog(
        [row.tariff_bdt_per_kwh for row in rows] + [0] * 96,
        A_eq=matrix, b_eq=rhs, bounds=bounds, method="highs",
    )


@pytest.mark.parametrize("seed", range(60))
def test_randomized_differential(seed):
    rng = np.random.default_rng(seed)
    request = ScenarioRequest(
        scenario_id=f"random-{seed}",
        operator_notes=["Synthetic directive"] * 3,
        hours=[
            dict(hour=h, demand_kwh=float(rng.uniform(0, 30)),
                 solar_kwh=float(rng.uniform(0, 45)), tariff_bdt_per_kwh=float(rng.uniform(0, 30)))
            for h in range(24)
        ],
        battery=dict(capacity_kwh=120, initial_energy_kwh=60, minimum_energy_kwh=10,
                     max_charge_kwh_per_hour=20, max_discharge_kwh_per_hour=17),
    )
    types = ["solar_reduction", "minimum_battery_reserve", "no_charge_window",
             "no_discharge_window", "max_grid_window"]
    raw = []
    for index in range(3):
        kind = types[(seed + index) % len(types)]
        start = int(rng.integers(0, 22))
        adjustment = {"hours": list(range(start, min(24, start + 3)))}
        if kind == "solar_reduction":
            adjustment["factor"] = float(rng.uniform(0, 1))
        elif kind == "minimum_battery_reserve":
            adjustment["minimum_energy_kwh"] = float(rng.uniform(20, 90))
        elif kind == "max_grid_window":
            adjustment["max_grid_kwh"] = float(rng.uniform(0, 30))
        raw.append(dict(note_index=index, applies=True, directive_type=kind,
                        structured_adjustment=adjustment, explanation="Synthetic test"))
    directives = validate_all(raw, 3, 120)
    oracle = reference_120_variable_lp(request, directives)
    if oracle.status == 2:
        with pytest.raises(OptimizationError):
            solve(request.hours, request.battery, directives)
    else:
        assert oracle.success
        plan = solve(request.hours, request.battery, directives)
        _, cost, _ = replay(request.hours, request.battery, directives, plan)
        assert cost == pytest.approx(oracle.fun, abs=1e-6)


@pytest.mark.parametrize("solar,rate,tariff", [(0, 0, 5), (100, 0, 0), (100, 50, 0), (0, 50, 5)])
def test_flat_tariffs_surplus_and_zero_rates(solar, rate, tariff):
    request, _ = prepare(CASES[0])
    request.battery.max_charge_kwh_per_hour = rate
    request.battery.max_discharge_kwh_per_hour = rate
    for row in request.hours:
        row.demand_kwh, row.solar_kwh, row.tariff_bdt_per_kwh = 10, solar, tariff
    plan = solve(list(reversed(request.hours)), request.battery, [])
    _, cost, _ = replay(request.hours, request.battery, [], plan)
    assert cost == pytest.approx(24 * max(0, 10-solar) * tariff)


def test_final_hour_reserve_cannot_override_neutrality():
    request, _ = prepare(CASES[0])
    directives = validate_all([dict(
        note_index=0, applies=True, directive_type="minimum_battery_reserve",
        structured_adjustment={"hours": [23], "minimum_energy_kwh": 120},
        explanation="Reserve above initial energy at the terminal hour",
    )], 1, request.battery.capacity_kwh)
    with pytest.raises(OptimizationError):
        solve(request.hours, request.battery, directives)


def test_replay_catches_directive_compiler_bug(monkeypatch):
    request, directives = prepare(CASES[0])
    compiler = optimizer.apply_directives
    monkeypatch.setattr(optimizer, "apply_directives", lambda hours, battery, notes: compiler(hours, battery, []))
    plan = solve(request.hours, request.battery, directives)
    with pytest.raises(ReplayError):
        replay(request.hours, request.battery, directives, plan)


@pytest.mark.parametrize("corruption", ["nan", "balance", "state", "idle"])
def test_replay_rejects_tampered_plan(corruption):
    request, directives = prepare(CASES[0])
    plan = solve(request.hours, request.battery, directives)
    if corruption == "nan":
        plan[0] = plan[0].model_copy(update={"grid_kwh": float("nan")})
    elif corruption == "balance":
        plan[0] = plan[0].model_copy(update={"grid_kwh": plan[0].grid_kwh + 1})
    elif corruption == "state":
        plan[0] = plan[0].model_copy(update={"battery_energy_after_kwh": 10000})
    else:
        plan[0] = plan[0].model_copy(update={"battery_action": "idle", "battery_kwh": 1})
    with pytest.raises(ReplayError):
        replay(request.hours, request.battery, directives, plan)


def test_solver_failure_never_emits_plan(monkeypatch):
    request, directives = prepare(CASES[0])
    monkeypatch.setattr(optimizer, "linprog", lambda *a, **k: SimpleNamespace(success=False, x=None))
    with pytest.raises(OptimizationError):
        solve(request.hours, request.battery, directives)


def test_concurrent_requests_do_not_share_mutable_bounds():
    def run(case):
        request, directives = prepare(case)
        plan = solve(request.hours, request.battery, directives)
        return replay(request.hours, request.battery, directives, plan)[1]
    with ThreadPoolExecutor(max_workers=3) as executor:
        costs = list(executor.map(run, CASES))
    assert costs == pytest.approx([c["expected_output"]["total_cost_bdt"] for c in CASES], abs=0.01)
