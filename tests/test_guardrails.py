"""Section 08 guardrails, exhaustively. No LLM, no solver: malformed model output goes straight into
guardrails.validate_all() and MUST raise GuardrailError (never be accepted, coerced, or crash differently).

Cases marked BUG_* are real defects found by this matrix; they are xfail(strict) so the suite documents them
and turns red the moment they are fixed (remove the marker then).
"""
import copy
import math

import pytest

from app.guardrails import GuardrailError, validate_all

CAP = 200.0
NAN, INF = float("nan"), float("inf")

VALID = {
    "solar_reduction": {"hours": [13, 14], "factor": 0.2},
    "minimum_battery_reserve": {"hours": [18, 19], "minimum_energy_kwh": 120},
    "no_charge_window": {"hours": [2, 3]},
    "no_discharge_window": {"hours": [18]},
    "max_grid_window": {"hours": [18, 19], "max_grid_kwh": 155},
}


def good(i, dtype="no_charge_window"):
    if dtype == "no_op":
        return {"note_index": i, "applies": False, "directive_type": "no_op",
                "structured_adjustment": None, "explanation": "x"}
    return {"note_index": i, "applies": True, "directive_type": dtype,
            "structured_adjustment": copy.deepcopy(VALID[dtype]), "explanation": "x"}


def one(**over):
    """A single-note output for a directive, with overrides on the entry."""
    e = good(0)
    e.update(over)
    return [e]


def adj(dtype, **over):
    e = good(0, dtype)
    e["structured_adjustment"].update(over)
    return [e]


def drop_key(dtype, key):
    e = good(0, dtype)
    del e["structured_adjustment"][key]
    return [e]


def run(raw, n=1):
    return validate_all(raw, num_notes=n, battery_capacity=CAP)


# ------------------------------------------------------------------ positive controls (guard against over-rejecting)
@pytest.mark.parametrize("dtype", list(VALID) + ["no_op"])
def test_valid_entry_accepted_and_not_coerced(dtype):
    raw = [good(0, dtype)]
    out = run(raw)
    assert len(out) == 1 and out[0].directive_type == dtype
    assert out[0].structured_adjustment == raw[0]["structured_adjustment"]


@pytest.mark.parametrize("raw", [
    adj("solar_reduction", factor=0), adj("solar_reduction", factor=1), adj("solar_reduction", factor=0.0),
    adj("solar_reduction", hours=[0]), adj("solar_reduction", hours=[23]),
    adj("solar_reduction", hours=list(range(24))),
    adj("minimum_battery_reserve", minimum_energy_kwh=0),
    adj("minimum_battery_reserve", minimum_energy_kwh=CAP),
    adj("max_grid_window", max_grid_kwh=0), adj("max_grid_window", max_grid_kwh=1e9),
], ids=lambda r: str(r[0]["structured_adjustment"]))
def test_boundary_values_are_accepted(raw):
    run(raw)


def test_three_notes_mixed_with_noops():
    out = run([good(0, "solar_reduction"), good(1, "no_op"), good(2, "max_grid_window")], n=3)
    assert [d.note_index for d in out] == [0, 1, 2]


# ------------------------------------------------------------------ top level
@pytest.mark.parametrize("raw", [{}, {"note_index": 0}, "[]", None, 5, 1.5, True], ids=repr)
def test_top_level_not_a_list(raw):
    with pytest.raises(GuardrailError):
        run(raw)


@pytest.mark.parametrize("raw,n", [
    ([], 1), ([good(0)], 2), ([good(0), good(1)], 1), ([good(0)] * 4, 3), ([], 3),
], ids=["empty-1", "too-few", "too-many", "four-for-three", "empty-3"])
def test_wrong_length(raw, n):
    with pytest.raises(GuardrailError):
        run(raw, n)


@pytest.mark.parametrize("entry", [None, "no_op", 5, [good(0)], True], ids=repr)
def test_entry_not_an_object(entry):
    with pytest.raises(GuardrailError):
        run([entry])


# ------------------------------------------------------------------ note_index
@pytest.mark.parametrize("raw,n", [
    ([good(1), good(0)], 2),  # out of order
    ([good(0), good(0)], 2),  # duplicate
    ([good(1), good(2)], 2),  # starts at 1
    ([good(0), good(2), good(2)], 3),  # gap + duplicate
    ([good(0), good(2)], 2),  # gap
    ([good(-1)], 1),
    ([good(5)], 1),
], ids=["swapped", "duplicate", "starts-at-1", "gap+dup", "gap", "negative", "out-of-range"])
def test_bad_note_index_set(raw, n):
    with pytest.raises(GuardrailError):
        run(raw, n)


@pytest.mark.parametrize("bad", [None, "0", [0], {}], ids=repr)
def test_note_index_wrong_type(bad):
    with pytest.raises(GuardrailError):
        run(one(note_index=bad))


def test_note_index_missing():
    e = good(0)
    del e["note_index"]
    with pytest.raises(GuardrailError):
        run([e])


@pytest.mark.xfail(strict=True, reason="BUG_INDEX_COERCION: `0.0 != 0` and `True != 1` are False in Python, so "
                                       "float/bool note_index values are silently accepted")
@pytest.mark.parametrize("raw,n", [
    ([good(0), {**good(1), "note_index": True}], 2),
    ([{**good(0), "note_index": 0.0}], 1),
    ([{**good(0), "note_index": False}], 1),
], ids=["True-for-1", "0.0-for-0", "False-for-0"])
def test_note_index_bool_or_float_must_not_be_coerced(raw, n):
    with pytest.raises(GuardrailError):
        run(raw, n)


# ------------------------------------------------------------------ directive_type
@pytest.mark.parametrize("bad", [
    "battery_boost", "", " ", "SOLAR_REDUCTION", "solar_reduction ", "solar-reduction", "NO_OP", "noop", "none", None, 5,
], ids=repr)
def test_directive_type_outside_allowed_values(bad):
    with pytest.raises(GuardrailError):
        run(one(directive_type=bad))


def test_directive_type_missing():
    e = good(0)
    del e["directive_type"]
    with pytest.raises(GuardrailError):
        run([e])


@pytest.mark.xfail(strict=True, reason="BUG_UNHASHABLE_TYPE: `list in set` raises TypeError, not GuardrailError, so "
                                       "the retry loop is bypassed and the exception escapes graph.invoke")
@pytest.mark.parametrize("bad", [["no_op"], {"a": 1}], ids=repr)
def test_directive_type_unhashable_raises_guardrail_error(bad):
    with pytest.raises(GuardrailError):
        run(one(directive_type=bad))


# ------------------------------------------------------------------ applies semantics
@pytest.mark.parametrize("bad", [False, None, "true", 1, 0, "yes"], ids=repr)
def test_non_noop_requires_applies_true(bad):
    with pytest.raises(GuardrailError):
        run(one(applies=bad))


def test_applies_missing_on_non_noop():
    e = good(0)
    del e["applies"]
    with pytest.raises(GuardrailError):
        run([e])


@pytest.mark.parametrize("bad", [True, None, "false", 0, 1], ids=repr)
def test_noop_requires_applies_false(bad):
    with pytest.raises(GuardrailError):
        run([{**good(0, "no_op"), "applies": bad}])


@pytest.mark.parametrize("adjv", [{}, {"hours": [1]}, [], "null", 0], ids=repr)
def test_noop_requires_null_adjustment(adjv):
    with pytest.raises(GuardrailError):
        run([{**good(0, "no_op"), "structured_adjustment": adjv}])


@pytest.mark.parametrize("dtype", VALID)
@pytest.mark.parametrize("bad", [None, [], "x", 5, [13, 14]], ids=repr)
def test_non_noop_requires_object_adjustment(dtype, bad):
    with pytest.raises(GuardrailError):
        run([{**good(0, dtype), "structured_adjustment": bad}])


# ------------------------------------------------------------------ keys, per directive type
@pytest.mark.parametrize("dtype", VALID)
def test_empty_adjustment_object(dtype):
    with pytest.raises(GuardrailError):
        run([{**good(0, dtype), "structured_adjustment": {}}])


@pytest.mark.parametrize("dtype,key", [(d, k) for d, a in VALID.items() for k in a])
def test_each_required_key_missing(dtype, key):
    with pytest.raises(GuardrailError):
        run(drop_key(dtype, key))


@pytest.mark.parametrize("dtype", VALID)
@pytest.mark.parametrize("extra", ["factor_", "note", "minimum_energy_kwh", "max_grid_kwh", "factor", "hours2"])
def test_extra_key_rejected(dtype, extra):
    if extra in VALID[dtype]:
        pytest.skip("that key is required for this type, not extra")
    with pytest.raises(GuardrailError):
        run(adj(dtype, **{extra: 1}))


@pytest.mark.parametrize("dtype,wrong_key", [
    ("no_charge_window", "factor"), ("no_discharge_window", "max_grid_kwh"),
    ("solar_reduction", "max_grid_kwh"), ("max_grid_window", "factor"),
    ("minimum_battery_reserve", "max_grid_kwh"),
])
def test_wrong_keys_for_type_swapped(dtype, wrong_key):
    e = good(0, dtype)
    e["structured_adjustment"] = {"hours": [1, 2], wrong_key: 5}
    with pytest.raises(GuardrailError):
        run([e])


# ------------------------------------------------------------------ hours (checked on every hours-bearing type)
BAD_HOURS = [
    13, "13", None, {"h": 13}, "[13]",  # non-list
    [],  # empty
    [13.0], [13.5], [True], [False], ["13"], [None], [[13]], [13, "14"],  # non-int members
    [-1], [24], [100], [13, 24], [-5, 3],  # out of range
    [13, 13], [1, 2, 2],  # duplicates
    [14, 13], [23, 0], [1, 3, 2],  # not ascending
]


@pytest.mark.parametrize("dtype", VALID)
@pytest.mark.parametrize("bad", BAD_HOURS, ids=repr)
def test_bad_hours(dtype, bad):
    with pytest.raises(GuardrailError):
        run(adj(dtype, hours=bad))


def test_hours_tuple_like_and_huge_ints():
    with pytest.raises(GuardrailError):
        run(adj("no_charge_window", hours=[10 ** 30]))


# ------------------------------------------------------------------ solar factor
@pytest.mark.parametrize("bad", [-0.1, -1, 1.0001, 1.1, 2, 100, "0.2", None, True, False, [0.2], {"f": 1}, NAN, INF, -INF],
                         ids=repr)
def test_bad_factor(bad):
    with pytest.raises(GuardrailError):
        run(adj("solar_reduction", factor=bad))


# ------------------------------------------------------------------ reserve
@pytest.mark.parametrize("bad", [-1, -0.001, CAP + 0.011, CAP + 1, 1e9, "100", None, True, [100], -INF],
                         ids=repr)
def test_bad_reserve(bad):
    with pytest.raises(GuardrailError):
        run(adj("minimum_battery_reserve", minimum_energy_kwh=bad))


@pytest.mark.xfail(strict=True, reason="BUG_NONFINITE_RESERVE: Section 08 requires finite reserve values; NaN "
                                       "passes both `val < 0` and `val > capacity` (comparisons with NaN are False)")
@pytest.mark.parametrize("bad", [NAN], ids=repr)
def test_nonfinite_reserve_rejected(bad):
    with pytest.raises(GuardrailError):
        run(adj("minimum_battery_reserve", minimum_energy_kwh=bad))


def test_inf_reserve_rejected_via_capacity_check():
    with pytest.raises(GuardrailError):
        run(adj("minimum_battery_reserve", minimum_energy_kwh=INF))


# ------------------------------------------------------------------ grid cap
@pytest.mark.parametrize("bad", [-1, -0.001, "155", None, True, [155], -INF], ids=repr)
def test_bad_max_grid(bad):
    with pytest.raises(GuardrailError):
        run(adj("max_grid_window", max_grid_kwh=bad))


@pytest.mark.xfail(strict=True, reason="BUG_NONFINITE_GRID_CAP: Section 08 requires max_grid_kwh to be finite; "
                                       "NaN and +inf both pass `val < 0` and are accepted")
@pytest.mark.parametrize("bad", [NAN, INF], ids=repr)
def test_nonfinite_grid_cap_rejected(bad):
    with pytest.raises(GuardrailError):
        run(adj("max_grid_window", max_grid_kwh=bad))


# ------------------------------------------------------------------ explanation (non-string must not crash)
@pytest.mark.xfail(strict=True, reason="BUG_EXPLANATION_TYPE: a non-string explanation makes pydantic raise "
                                       "ValidationError inside validate_entry, which is not a GuardrailError")
@pytest.mark.parametrize("bad", [123, ["a"], 1.5, True], ids=repr)
def test_non_string_explanation_is_a_guardrail_error(bad):
    with pytest.raises(GuardrailError):
        run(one(explanation=bad))


def test_missing_or_null_explanation_tolerated():
    e = good(0)
    del e["explanation"]
    assert run([e])[0].explanation == ""
    assert run(one(explanation=None))[0].explanation == ""


def test_finite_check_sanity():  # keeps the constants honest
    assert math.isnan(NAN) and math.isinf(INF)
