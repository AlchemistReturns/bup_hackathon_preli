"""The two edge cases that broke naive implementations: forced curtailment and infeasible directives."""
import pulp
import pytest

from app.optimizer import _build, compile_bounds, solve
from app.replay import ReplayError, replay
from helpers import check_physics, directive, make_battery, make_hours


def test_forced_curtailment_solar_used_sits_below_ceiling():
    # effective solar (300) >> demand (40) + max_charge (30) for hours 8-15: surplus has nowhere to go.
    demand = [40.0] * 24
    solar = [300.0 if 8 <= h <= 15 else 0.0 for h in range(24)]
    hours = make_hours(demand, solar, [6.0] * 24)
    bat = make_battery(cap=200, init=100, min_=40, chg=30, dis=30)

    plan, mode = solve(hours, bat, [])
    assert mode == "optimal"
    b = compile_bounds(hours, bat, [])
    assert check_physics(hours, bat, b.eff_solar, plan) == []
    replay(hours, bat, [], plan)

    for h in range(8, 16):
        p = plan[h]
        assert p.solar_used_kwh < b.eff_solar[h] - 100  # curtailed, not pinned to the ceiling
        assert p.solar_used_kwh <= demand[h] + bat.max_charge_kwh_per_hour + 1e-6
        assert p.grid_kwh >= 0


def test_pinning_solar_to_ceiling_would_be_infeasible():
    """Documents WHY solar_used is a free variable: forcing s == effective solar makes the model infeasible."""
    demand = [40.0] * 24
    solar = [300.0 if 8 <= h <= 15 else 0.0 for h in range(24)]
    hours = make_hours(demand, solar, [6.0] * 24)
    bat = make_battery(cap=200, init=100, min_=40, chg=30, dis=30)
    b = compile_bounds(hours, bat, [])

    prob, grid, s, *_ = _build(hours, bat, b, elastic=False)
    for h in range(8, 16):
        s[h].lowBound = s[h].upBound  # pin
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    assert pulp.LpStatus[prob.status] == "Infeasible"


def test_infeasible_directives_return_min_violation_plan_not_error():
    demand = [100.0] * 24
    hours = make_hours(demand, [0.0] * 24, [5.0] * 24)
    bat = make_battery(cap=200, init=200, min_=40, chg=50, dis=30)
    # Cap of 60 in 18-21 with demand 100, discharge <= 30/h, battery full at start and required full at end.
    # Day-end neutrality means net discharge over 18-23 <= 0, and only hours 22-23 remain to recharge
    # (2 x 50 kWh), so total discharge in 18-21 <= 100 and the minimum total excess is 4*40 - 100 = 60 kWh.
    ds = [directive("max_grid_window", hours=[18, 19, 20, 21], max_grid_kwh=60)]

    plan, mode = solve(hours, bat, ds)
    assert mode == "relaxed"
    b = compile_bounds(hours, bat, ds)
    assert check_physics(hours, bat, b.eff_solar, plan) == []
    excess = sum(plan[h].grid_kwh - 60 for h in (18, 19, 20, 21))
    assert excess == pytest.approx(60.0, abs=1e-4)  # minimum total violation (unrelaxed would be 160)

    with pytest.raises(ReplayError):  # replay enforces the cap, so the directive-aware replay must reject it
        replay(hours, bat, ds, plan)
    replay(hours, bat, [], plan)  # physics-only replay passes


def test_reserve_above_capacity_and_zero_cap_everywhere_still_return_a_plan():
    hours = make_hours([80.0] * 24, [0.0] * 24, [5.0] * 24)
    bat = make_battery(cap=100, init=60, min_=10, chg=20, dis=20)
    ds = [
        directive("minimum_battery_reserve", hours=list(range(24)), minimum_energy_kwh=100),
        directive("max_grid_window", hours=list(range(24)), max_grid_kwh=0),
    ]
    plan, mode = solve(hours, bat, ds)
    assert mode == "relaxed"
    assert check_physics(hours, bat, compile_bounds(hours, bat, ds).eff_solar, plan) == []


def test_feasible_directives_stay_optimal_and_respected():
    hours = make_hours([100.0] * 24, [0.0] * 24, [5.0] * 12 + [9.0] * 12)
    bat = make_battery(cap=200, init=100, min_=40, chg=50, dis=50)
    ds = [directive("minimum_battery_reserve", hours=[18, 19], minimum_energy_kwh=90),
          directive("no_charge_window", hours=[0, 1])]
    plan, mode = solve(hours, bat, ds)
    assert mode == "optimal"
    replay(hours, bat, ds, plan)
    assert plan[0].battery_action != "charge" and plan[1].battery_action != "charge"
    assert plan[18].battery_energy_after_kwh >= 90 - 1e-6
