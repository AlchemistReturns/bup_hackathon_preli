"""Randomised physics check: the optimizer must never emit a physically invalid plan, whatever the directives."""
import random

import pulp
import pytest

from app.optimizer import N, _build, compile_bounds, solve
from app.replay import replay
from helpers import check_physics, directive, make_battery, make_hours

SCENARIOS = 400
SEED = 20260918
# CBC's solution file carries ~8 significant digits, so with non-round data (this generator uses cap/3 rate
# limits on purpose) plan values can be off by ~1e-5. 1e-4 is 100x tighter than the judge's 0.01 replay
# tolerance. idle => battery_kwh == 0 and end-of-day energy == initial are checked exactly regardless.
PHYSICS_TOL = 1e-4


def _window(rng):
    start = rng.randrange(0, 23)
    return list(range(start, min(24, start + rng.randint(1, 8))))


def _random_directive(rng, battery, demand):
    t = rng.choice(["solar_reduction", "minimum_battery_reserve", "no_charge_window",
                    "no_discharge_window", "max_grid_window"])
    hrs = _window(rng)
    if t == "solar_reduction":
        return directive(t, hours=hrs, factor=round(rng.random(), 2))
    if t == "minimum_battery_reserve":
        return directive(t, hours=hrs, minimum_energy_kwh=round(rng.uniform(0, battery.capacity_kwh), 1))
    if t == "max_grid_window":
        return directive(t, hours=hrs, max_grid_kwh=round(rng.uniform(0, max(demand) * 1.2), 1))
    return directive(t, hours=hrs)


def _scenario(rng):
    demand = [round(rng.uniform(0, 200), 1) for _ in range(24)]
    solar_scale = rng.choice([0, 30, 150, 400])  # 400 forces curtailment in many hours
    solar = [round(rng.uniform(0, solar_scale), 1) if 6 <= h <= 18 else 0.0 for h in range(24)]
    tariff = [rng.choice([0, 0, 3, 5, 6, 9, 12, 15]) + round(rng.random(), 1) * rng.choice([0, 1]) for _ in range(24)]
    cap = rng.choice([50, 100, 200, 400])
    lo = round(rng.uniform(0, 0.4) * cap, 1)
    init = round(rng.uniform(lo, cap), 1)
    battery = make_battery(cap=cap, init=init, min_=lo,
                           chg=rng.choice([0, 10, 30, cap / 3, cap]), dis=rng.choice([0, 10, 30, cap / 3, cap]))
    hours = make_hours(demand, solar, tariff)
    directives = [_random_directive(rng, battery, demand) for _ in range(rng.randint(0, 3))]
    return hours, battery, directives


def test_fuzz_physics_never_violated():
    rng = random.Random(SEED)
    relaxed = 0
    problems = []
    for i in range(SCENARIOS):
        hours, battery, directives = _scenario(rng)
        plan, mode = solve(hours, battery, directives)
        bounds = compile_bounds(hours, battery, directives)

        errs = check_physics(hours, battery, bounds.eff_solar, plan, tol=PHYSICS_TOL)
        if errs:
            problems.append((i, mode, errs[:3]))
            continue

        if mode == "optimal":
            replay(hours, battery, directives, plan)  # full replay incl. directive limits must pass
            # When the hard model is feasible, the elastic model must return the same cost.
            prob, grid, *_ = _build(hours, battery, bounds, elastic=True)
            prob.solve(pulp.PULP_CBC_CMD(msg=0))
            tariff = {h.hour: h.tariff_bdt_per_kwh for h in hours}
            plan_cost = sum(p.grid_kwh * tariff[p.hour] for p in plan)
            elastic_cost = sum(grid[h].value() * tariff[h] for h in range(N))
            assert elastic_cost == pytest.approx(plan_cost, abs=0.05, rel=1e-6), f"scenario {i}"
        else:
            relaxed += 1
            replay(hours, battery, [], plan)  # physics-only replay must pass

    print(f"\nfuzz: {SCENARIOS} scenarios, physics violations={len(problems)}, elastic fallback={relaxed}")
    assert not problems, problems[:5]
    assert relaxed > 0, "generator never exercised the elastic path"
