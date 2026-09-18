"""Section 11.5: absolute tolerance of 0.01 kWh / 0.01 BDT. Every replay() check is probed at 0.009 (must pass) and
0.011 (must be rejected), so no comparison is stricter or looser than the spec. Plans are built by hand so that
each probe perturbs exactly ONE quantity."""
import pytest

import app.optimizer as optimizer
import app.replay as replay_mod
from app.replay import ReplayError, replay
from app.schemas import HourlyPlanEntry
from helpers import directive, make_battery, make_hours

DEMAND = 100.0
SOLAR = [50.0 if 8 <= h <= 15 else 0.0 for h in range(24)]
HOURS = make_hours([DEMAND] * 24, SOLAR, [5.0] * 24)
BAT = make_battery(cap=200, init=100, min_=40, chg=50, dis=50)

INSIDE, OUTSIDE = 0.009, 0.011


def build(actions=None, battery=BAT, overrides=None):
    """actions: {hour: (action, kwh, solar_used)}. Grid comes from the balance equation, energy from the chain."""
    actions, overrides = actions or {}, overrides or {}
    prev, rows = battery.initial_energy_kwh, []
    for h in range(24):
        action, kwh, solar = actions.get(h, ("idle", 0.0, 0.0))
        ch, dis = (kwh if action == "charge" else 0.0), (kwh if action == "discharge" else 0.0)
        prev = prev + ch - dis
        row = dict(hour=h, grid_kwh=DEMAND + ch - solar - dis, solar_used_kwh=solar, battery_action=action,
                   battery_kwh=kwh, battery_energy_after_kwh=prev)
        row.update(overrides.get(h, {}))
        rows.append(HourlyPlanEntry(**row))
    return rows


def probe(name, make_plan, directives=(), battery=BAT):
    """make_plan(m) -> plan whose only defect has size m. m=0 and 0.009 must replay; 0.011 and 0.05 must not."""
    ds = list(directives)
    for m in (0.0, INSIDE):
        replay(HOURS, battery, ds, make_plan(m))
    for m in (OUTSIDE, 0.05):
        with pytest.raises(ReplayError):
            replay(HOURS, battery, ds, make_plan(m))


def test_replay_tolerance_constant_is_exactly_0_01():
    assert replay_mod.TOL == 0.01


def test_solar_used_vs_effective_solar():
    probe("solar>raw", lambda m: build({9: ("idle", 0.0, 50.0 + m)}))


def test_solar_used_vs_solar_after_reduction_directive():
    ds = [directive("solar_reduction", hours=[9], factor=0.5)]  # effective 25, raw 50
    probe("solar>effective", lambda m: build({9: ("idle", 0.0, 25.0 + m)}), ds)


def test_charge_rate_limit():
    probe("charge>max", lambda m: build({10: ("charge", 50.0 + m, 0.0), 11: ("discharge", 50.0, 0.0),
                                        12: ("discharge", m, 0.0)}))


def test_discharge_rate_limit():
    probe("discharge>max", lambda m: build({10: ("discharge", 50.0 + m, 0.0), 11: ("charge", 50.0, 0.0),
                                           12: ("charge", m, 0.0)}))


def test_energy_chain_must_match_action():
    probe("chain", lambda m: build(overrides={5: {"battery_energy_after_kwh": 100.0 + m}}))


def test_energy_upper_bound_capacity():
    full = make_battery(cap=200, init=200, min_=40, chg=50, dis=50)
    probe("cap", lambda m: build({10: ("charge", m, 0.0), 11: ("discharge", m, 0.0)}, battery=full), battery=full)


def test_energy_lower_bound_base_minimum():
    low = make_battery(cap=200, init=40, min_=40, chg=50, dis=50)
    probe("min", lambda m: build({10: ("discharge", m, 0.0), 11: ("charge", m, 0.0)}, battery=low), battery=low)


def test_reserve_directive_floor():
    ds = [directive("minimum_battery_reserve", hours=[18], minimum_energy_kwh=90)]
    probe("reserve", lambda m: build({17: ("discharge", 10.0 + m, 0.0), 20: ("charge", 10.0 + m, 0.0)}), ds)


def test_grid_cap_directive():
    ds = [directive("max_grid_window", hours=[18], max_grid_kwh=80)]
    probe("cap", lambda m: build({18: ("discharge", 20.0 - m, 0.0), 19: ("charge", 20.0 - m, 0.0)}), ds)


def test_energy_balance():
    probe("balance", lambda m: build(overrides={5: {"grid_kwh": DEMAND + m}}))


def test_end_of_day_neutrality():
    probe("neutrality", lambda m: build({23: ("charge", m, 0.0)}))


def test_no_charge_window_allows_only_tolerance_sized_charge():
    ds = [directive("no_charge_window", hours=[2])]
    probe("no_charge", lambda m: build({2: ("charge", m, 0.0), 4: ("discharge", m, 0.0)}), ds)


def test_no_discharge_window_allows_only_tolerance_sized_discharge():
    ds = [directive("no_discharge_window", hours=[3])]
    probe("no_discharge", lambda m: build({3: ("discharge", m, 0.0), 5: ("charge", m, 0.0)}), ds)


# ------------------------------------------------------------------ constants elsewhere in the codebase
def test_our_own_output_precision_is_much_tighter_than_the_tolerance():
    for name in ("ZERO", "COST_TOL", "STAGE2_SLACK"):
        assert getattr(optimizer, name) < 0.01 / 5, name
    assert 10 ** -optimizer.DP < 0.01 / 1000  # plan values are rounded far below the tolerance


@pytest.mark.xfail(strict=True, reason="TOL_GUARDRAIL_RESERVE: guardrails compare `minimum_energy_kwh > capacity` "
                                       "exactly. Section 11.5 says values within 0.01 are equivalent, so a reserve "
                                       "of capacity + float noise (e.g. 100% of capacity computed with rounding) "
                                       "is rejected, retried, and finally safe-failed to no_op")
@pytest.mark.parametrize("excess", [1e-12, 1e-9, 0.004])
def test_reserve_within_tolerance_of_capacity_is_accepted(excess):
    from app.guardrails import validate_all
    raw = [{"note_index": 0, "applies": True, "directive_type": "minimum_battery_reserve",
            "structured_adjustment": {"hours": [18], "minimum_energy_kwh": 200.0 + excess}, "explanation": "x"}]
    validate_all(raw, 1, 200.0)
