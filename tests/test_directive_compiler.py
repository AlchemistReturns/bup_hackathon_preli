"""Solver-free tests of apply_directives()/compile_bounds(): the "interpreted correctly but not applied" surface."""
from app.optimizer import apply_directives, compile_bounds
from app.schemas import DirectiveInterpretation
from helpers import directive, make_battery, make_hours

HOURS = make_hours([100] * 24, [80] * 24, [5] * 24)
BAT = make_battery(cap=200, init=100, min_=40, chg=50, dis=45)


def test_reserve_applies_inside_window_only():
    solar, reserve, nc, nd, cap = apply_directives(
        HOURS, BAT, [directive("minimum_battery_reserve", hours=[18, 19, 20, 21], minimum_energy_kwh=90)]
    )
    assert [reserve[h] for h in (18, 19, 20, 21)] == [90, 90, 90, 90]
    assert all(reserve[h] == 40 for h in range(24) if h not in (18, 19, 20, 21))
    assert reserve[17] == 40 and reserve[22] == 40


def test_reserve_takes_max_never_lowers_and_never_last_wins():
    ds = [
        directive("minimum_battery_reserve", hours=[10, 11], minimum_energy_kwh=120),
        directive("minimum_battery_reserve", hours=[11, 12], minimum_energy_kwh=70),  # later, lower
        directive("minimum_battery_reserve", hours=[13], minimum_energy_kwh=10),  # below base minimum
    ]
    _, reserve, *_ = apply_directives(HOURS, BAT, ds)
    assert (reserve[10], reserve[11], reserve[12]) == (120, 120, 70)
    assert reserve[13] == 40


def test_overlapping_grid_caps_take_minimum_in_either_order():
    a = directive("max_grid_window", hours=[18, 19, 20], max_grid_kwh=155)
    b = directive("max_grid_window", hours=[19, 20, 21], max_grid_kwh=120)
    for ds in ([a, b], [b, a]):
        *_, cap = apply_directives(HOURS, BAT, ds)
        assert cap[18] == 155
        assert cap[19] == 120 and cap[20] == 120
        assert cap[21] == 120
        assert cap[17] is None and cap[22] is None


def test_overlapping_solar_reductions_multiply():
    ds = [
        directive("solar_reduction", hours=[12, 13], factor=0.5),
        directive("solar_reduction", hours=[13, 14], factor=0.4),
    ]
    solar, *_ = apply_directives(HOURS, BAT, ds)
    assert solar[12] == 40.0
    assert abs(solar[13] - 80 * 0.5 * 0.4) < 1e-12
    assert solar[14] == 32.0
    assert solar[11] == 80.0


def test_no_charge_window_zeroes_charge_bound_only():
    b = compile_bounds(HOURS, BAT, [directive("no_charge_window", hours=[2, 3, 4])])
    assert [b.chg_ub[h] for h in (2, 3, 4)] == [0.0, 0.0, 0.0]
    assert b.chg_ub[1] == 50 and b.chg_ub[5] == 50
    assert all(b.dis_ub[h] == 45 for h in range(24))  # discharge untouched


def test_no_discharge_window_zeroes_discharge_bound_only():
    b = compile_bounds(HOURS, BAT, [directive("no_discharge_window", hours=[18, 19])])
    assert [b.dis_ub[h] for h in (18, 19)] == [0.0, 0.0]
    assert b.dis_ub[17] == 45 and b.dis_ub[20] == 45
    assert all(b.chg_ub[h] == 50 for h in range(24))  # charge untouched


def test_window_sets_match_replay_contract():
    """replay.py tests `h in no_charge_hours`; these slots must be sets of only the affected hours."""
    _, _, nc, nd, _ = apply_directives(
        HOURS, BAT,
        [directive("no_charge_window", hours=[2, 3]), directive("no_discharge_window", hours=[18])],
    )
    assert nc == {2, 3} and nd == {18}


def test_no_op_is_ignored():
    noop = DirectiveInterpretation(note_index=0, applies=False, directive_type="no_op", structured_adjustment=None)
    solar, reserve, nc, nd, cap = apply_directives(HOURS, BAT, [noop])
    assert solar == {h: 80.0 for h in range(24)}
    assert set(reserve.values()) == {40.0}
    assert nc == set() and nd == set() and set(cap.values()) == {None}


def test_directives_do_not_mutate_inputs():
    before = [h.model_copy() for h in HOURS]
    apply_directives(HOURS, BAT, [directive("solar_reduction", hours=[12], factor=0.0)])
    assert HOURS == before
