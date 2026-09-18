"""Section 09 boundary conditions the public samples never touch. Every scenario: full directive-aware replay()
passes, an independent physics check passes, and the plan is checked against a hand-derived optimum or an
intuitive property (never just "did not crash")."""
import pytest

from app.optimizer import compile_bounds, plan_cost, solve
from app.replay import replay
from helpers import check_physics, directive, make_battery, make_hours

TOL = 1e-4


def run(hours, bat, ds=()):
    ds = list(ds)
    plan, mode = solve(hours, bat, ds)
    assert check_physics(hours, bat, compile_bounds(hours, bat, ds).eff_solar, plan, tol=TOL) == []
    if mode == "optimal":
        replay(hours, bat, ds, plan)
    return plan, mode


def flat(v):
    return [float(v)] * 24


def two_level(first, second, split=12):
    return [float(first)] * split + [float(second)] * (24 - split)


D100 = flat(100)
ZERO24 = flat(0)


# ------------------------------------------------------------------ initial energy at the rails
def test_full_battery_at_start_expensive_first_discharges_then_refills():
    # tariff 10 for h0-11, 5 after. Battery starts FULL (no charge headroom) and must end full.
    # Best: discharge 160 (=200-40) early at price 10, refill later at price 5 -> saves 160*5 = 800.
    hours = make_hours(D100, ZERO24, two_level(10, 5))
    bat = make_battery(cap=200, init=200, min_=40, chg=50, dis=50)
    plan, mode = run(hours, bat)
    assert mode == "optimal"
    assert plan_cost(hours, plan) == pytest.approx(100 * 10 * 12 + 100 * 5 * 12 - 800, abs=1e-3)
    assert plan[0].battery_action == "discharge"  # cannot charge at hour 0: no headroom
    assert plan[-1].battery_energy_after_kwh == 200.0
    assert all(p.battery_action != "charge" for p in plan[:12])  # refilling only when cheap


def test_full_battery_cheap_first_has_no_arbitrage_available():
    # tariff 5 then 10 with a full battery: nothing to charge cheaply, discharge-then-refill only loses. Cost = baseline.
    hours = make_hours(D100, ZERO24, two_level(5, 10))
    bat = make_battery(cap=200, init=200, min_=40, chg=50, dis=50)
    plan, _ = run(hours, bat)
    assert plan_cost(hours, plan) == pytest.approx(100 * 5 * 12 + 100 * 10 * 12, abs=1e-3)


def test_empty_battery_at_start_cheap_first_charges_then_discharges():
    # init == minimum: no discharge headroom. Charge 160 at price 5, use it at price 10 -> saves 800.
    hours = make_hours(D100, ZERO24, two_level(5, 10))
    bat = make_battery(cap=200, init=40, min_=40, chg=50, dis=50)
    plan, _ = run(hours, bat)
    assert plan_cost(hours, plan) == pytest.approx(100 * 5 * 12 + 100 * 10 * 12 - 800, abs=1e-3)
    assert plan[0].battery_action == "charge"  # cannot discharge at hour 0: at the floor
    assert plan[-1].battery_energy_after_kwh == 40.0
    assert min(p.battery_energy_after_kwh for p in plan) >= 40.0 - TOL


def test_empty_battery_expensive_first_has_no_arbitrage_available():
    hours = make_hours(D100, ZERO24, two_level(10, 5))
    bat = make_battery(cap=200, init=40, min_=40, chg=50, dis=50)
    plan, _ = run(hours, bat)
    assert plan_cost(hours, plan) == pytest.approx(100 * 10 * 12 + 100 * 5 * 12, abs=1e-3)


def test_battery_pinned_min_equals_capacity_equals_initial_is_all_idle():
    hours = make_hours(D100, flat(30), two_level(10, 5))
    bat = make_battery(cap=200, init=200, min_=200, chg=50, dis=50)
    plan, _ = run(hours, bat)
    assert all(p.battery_action == "idle" and p.battery_kwh == 0.0 for p in plan)


# ------------------------------------------------------------------ zero rate limits
def baseline_cost(hours):
    return sum(max(0.0, h.demand_kwh - h.solar_kwh) * h.tariff_bdt_per_kwh for h in hours)


@pytest.mark.parametrize("chg,dis", [(0, 50), (50, 0), (0, 0)], ids=["no-charge-rate", "no-discharge-rate", "both-zero"])
def test_zero_rate_battery_cannot_move_energy_so_it_stays_idle(chg, dis):
    # With one direction impossible, end-of-day neutrality forces the other to zero too.
    solar = [0.0] * 8 + [60.0] * 8 + [0.0] * 8
    hours = make_hours(D100, solar, two_level(5, 12))
    bat = make_battery(cap=200, init=100, min_=40, chg=chg, dis=dis)
    plan, mode = run(hours, bat)
    assert mode == "optimal"
    assert all(p.battery_action == "idle" and p.battery_kwh == 0.0 for p in plan)
    assert all(p.battery_energy_after_kwh == 100.0 for p in plan)
    assert plan_cost(hours, plan) == pytest.approx(baseline_cost(hours), abs=1e-3)  # solar still used fully


# ------------------------------------------------------------------ solar swamps everything
def test_solar_far_above_demand_plus_charge_for_all_24_hours():
    hours = make_hours(flat(50), flat(500), two_level(9, 12))
    bat = make_battery(cap=200, init=100, min_=40, chg=30, dis=30)
    plan, mode = run(hours, bat)
    assert mode == "optimal"
    assert plan_cost(hours, plan) == 0.0 and all(p.grid_kwh == 0.0 for p in plan)
    for p in plan:
        assert p.solar_used_kwh <= 50 + 30 + TOL  # can only use demand + charge headroom
        assert p.solar_used_kwh < 500 - 300  # the rest is curtailed, not forced through the balance
    assert plan[-1].battery_energy_after_kwh == 100.0


def test_solar_swamp_with_solar_reduction_on_every_hour_still_feasible():
    hours = make_hours(flat(50), flat(500), flat(9))
    bat = make_battery(cap=200, init=100, min_=40, chg=30, dis=30)
    plan, mode = run(hours, bat, [directive("solar_reduction", hours=list(range(24)), factor=0.05)])  # 25 kWh/h
    assert mode == "optimal"
    b = compile_bounds(hours, bat, [directive("solar_reduction", hours=list(range(24)), factor=0.05)])
    assert all(v == pytest.approx(25.0) for v in b.eff_solar.values())
    # 25 kWh solar < 50 demand: grid must cover the rest (battery nets zero over the day) -> cost = 25 * 24 * 9
    assert plan_cost(hours, plan) == pytest.approx(25 * 24 * 9, abs=1e-3)


def test_zero_solar_factor_removes_solar_entirely():
    hours = make_hours(D100, flat(80), flat(6))
    bat = make_battery(cap=200, init=100, min_=40, chg=50, dis=50)
    ds = [directive("solar_reduction", hours=list(range(24)), factor=0)]
    plan, _ = run(hours, bat, ds)
    assert all(p.solar_used_kwh == 0.0 for p in plan)
    assert plan_cost(hours, plan) == pytest.approx(100 * 24 * 6, abs=1e-3)


# ------------------------------------------------------------------ directives stacked on the same hours
def test_reserve_and_grid_cap_on_identical_hours_tight_but_feasible():
    # Hours 18-21: demand 100, cap 60 => must discharge exactly 40/h (160 total); reserve 90 => e>=90 after each.
    # Battery capacity 250 => e[17] <= 250, and e[17] - 160 >= 90 => e[17] == 250 exactly: zero slack anywhere.
    hours = make_hours(D100, ZERO24, flat(5))
    bat = make_battery(cap=250, init=200, min_=40, chg=60, dis=60)
    ds = [directive("minimum_battery_reserve", hours=[18, 19, 20, 21], minimum_energy_kwh=90),
          directive("max_grid_window", hours=[18, 19, 20, 21], max_grid_kwh=60)]
    plan, mode = run(hours, bat, ds)
    assert mode == "optimal"
    assert plan[17].battery_energy_after_kwh == pytest.approx(250.0, abs=TOL)
    assert plan[21].battery_energy_after_kwh == pytest.approx(90.0, abs=TOL)
    for h in (18, 19, 20, 21):
        assert plan[h].grid_kwh == pytest.approx(60.0, abs=TOL)
        assert plan[h].battery_action == "discharge" and plan[h].battery_kwh == pytest.approx(40.0, abs=TOL)


def test_reserve_and_cap_one_kwh_past_feasible_falls_to_relaxed_not_error():
    hours = make_hours(D100, ZERO24, flat(5))
    bat = make_battery(cap=249, init=200, min_=40, chg=60, dis=60)  # 1 kWh short of the tight case above
    ds = [directive("minimum_battery_reserve", hours=[18, 19, 20, 21], minimum_energy_kwh=90),
          directive("max_grid_window", hours=[18, 19, 20, 21], max_grid_kwh=60)]
    plan, mode = run(hours, bat, ds)
    assert mode == "relaxed"
    replay(hours, bat, [], plan)


def test_overlapping_windows_of_different_types_on_the_same_hours():
    hours = make_hours(D100, flat(20), two_level(4, 9))
    bat = make_battery(cap=300, init=150, min_=40, chg=60, dis=60)
    ds = [directive("no_charge_window", hours=list(range(10, 20))),
          directive("no_discharge_window", hours=list(range(15, 24))),
          directive("minimum_battery_reserve", hours=list(range(12, 22)), minimum_energy_kwh=120)]
    plan, mode = run(hours, bat, ds)
    assert mode == "optimal"
    for h in range(10, 20):
        assert plan[h].battery_action != "charge"
    for h in range(15, 24):
        assert plan[h].battery_action != "discharge"
    for h in range(12, 22):
        assert plan[h].battery_energy_after_kwh >= 120 - TOL


# ------------------------------------------------------------------ every hour carries a directive
def _demand_profile():
    return [100.0 + (h % 6) * 5 for h in range(24)]


@pytest.mark.parametrize("ds_name,ds", [
    ("nocharge/gridcap/reserve", [
        directive("no_charge_window", hours=list(range(0, 8))),
        directive("max_grid_window", hours=list(range(8, 16)), max_grid_kwh=140),
        directive("minimum_battery_reserve", hours=list(range(16, 24)), minimum_energy_kwh=100)]),
    ("nodischarge/solar/gridcap", [
        directive("no_discharge_window", hours=list(range(0, 8))),
        directive("solar_reduction", hours=list(range(8, 16)), factor=0.3),
        directive("max_grid_window", hours=list(range(16, 24)), max_grid_kwh=135)]),
    ("reserve/nocharge/nodischarge", [
        directive("minimum_battery_reserve", hours=list(range(0, 8)), minimum_energy_kwh=110),
        directive("no_charge_window", hours=list(range(8, 16))),
        directive("no_discharge_window", hours=list(range(16, 24)))]),
], ids=lambda x: x if isinstance(x, str) else "")
def test_every_hour_covered_by_a_directive(ds_name, ds):
    solar = [0.0] * 7 + [40.0, 80.0, 120.0, 140.0, 140.0, 120.0, 80.0, 40.0] + [0.0] * 9
    hours = make_hours(_demand_profile(), solar, [4.0] * 8 + [7.0] * 8 + [11.0] * 8)
    bat = make_battery(cap=300, init=150, min_=40, chg=60, dis=60)
    plan, mode = run(hours, bat, ds)
    assert mode == "optimal", ds_name
    free, _ = solve(hours, bat, [])
    assert plan_cost(hours, plan) >= plan_cost(hours, free) - 1e-6  # constraints can only cost, never save
    # the directive limits themselves are re-verified by replay() inside run()
    assert plan[-1].battery_energy_after_kwh == 150.0


# ------------------------------------------------------------------ degenerate objective (regression: HTTP 500 in stage 2)
@pytest.mark.parametrize("name,demand,solar,tariff", [
    ("all-zero-tariff", D100, ZERO24, ZERO24),
    ("everything-zero", ZERO24, ZERO24, ZERO24),
    ("zero-demand-with-solar", ZERO24, flat(40), flat(7)),
], ids=lambda x: x if isinstance(x, str) else "")
def test_degenerate_objective_still_returns_an_optimal_plan(name, demand, solar, tariff):
    hours = make_hours(demand, solar, tariff)
    bat = make_battery(cap=200, init=100, min_=40, chg=50, dis=50)
    plan, mode = run(hours, bat)
    assert mode == "optimal" and plan_cost(hours, plan) == 0.0
