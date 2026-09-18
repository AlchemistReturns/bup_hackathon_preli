"""
Deterministic validation of the LLM's raw directive output.
The LLM's JSON is never trusted directly — every field is checked here
before anything is handed to the optimizer. Any failure raises
GuardrailError with a message that is fed back to the LLM for a retry.
"""
import math
from typing import Any, Dict, List
from app.config import DIRECTIVE_TYPES
from app.schemas import DirectiveInterpretation


class GuardrailError(Exception):
    pass


REQUIRED_KEYS = {
    "solar_reduction": {"hours", "factor"},
    "minimum_battery_reserve": {"hours", "minimum_energy_kwh"},
    "no_charge_window": {"hours"},
    "no_discharge_window": {"hours"},
    "max_grid_window": {"hours", "max_grid_kwh"},
}


def _check_hours(hours: Any) -> List[int]:
    if not isinstance(hours, list) or len(hours) == 0:
        raise GuardrailError("hours must be a non-empty list of integers")
    if any(not isinstance(h, int) or isinstance(h, bool) for h in hours):
        raise GuardrailError("hours must contain integers only")
    if any(h < 0 or h > 23 for h in hours):
        raise GuardrailError("hours must be within 0-23")
    if len(set(hours)) != len(hours):
        raise GuardrailError("hours must not contain duplicates")
    if hours != sorted(hours):
        raise GuardrailError("hours must be in ascending order")
    return hours


def validate_entry(entry: Dict[str, Any], expected_note_index: int, battery_capacity: float) -> DirectiveInterpretation:
    if not isinstance(entry, dict):
        raise GuardrailError(f"entry for note {expected_note_index} is not an object")

    note_index = entry.get("note_index")
    if type(note_index) is not int or note_index != expected_note_index:
        raise GuardrailError(
            f"expected note_index {expected_note_index} at this position, got {note_index!r}"
        )

    directive_type = entry.get("directive_type")
    if not isinstance(directive_type, str) or directive_type not in DIRECTIVE_TYPES:
        raise GuardrailError(
            f"note {expected_note_index}: directive_type must be one of {sorted(DIRECTIVE_TYPES)}, got {directive_type!r}"
        )

    applies = entry.get("applies")
    adjustment = entry.get("structured_adjustment")
    explanation = entry.get("explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        raise GuardrailError(f"note {expected_note_index}: explanation must be a non-empty string")
    if "structured_adjustment" not in entry:
        raise GuardrailError(f"note {expected_note_index}: structured_adjustment is required")

    if directive_type == "no_op":
        if applies is not False:
            raise GuardrailError(f"note {expected_note_index}: no_op requires applies=false")
        if adjustment is not None:
            raise GuardrailError(f"note {expected_note_index}: no_op requires structured_adjustment=null")
        return DirectiveInterpretation(
            note_index=note_index,
            applies=False,
            directive_type="no_op",
            structured_adjustment=None,
            explanation=explanation,
        )

    if applies is not True:
        raise GuardrailError(f"note {expected_note_index}: {directive_type} requires applies=true")
    if not isinstance(adjustment, dict):
        raise GuardrailError(f"note {expected_note_index}: {directive_type} requires a structured_adjustment object")

    required = REQUIRED_KEYS[directive_type]
    missing = required - adjustment.keys()
    if missing:
        raise GuardrailError(f"note {expected_note_index}: {directive_type} missing keys {sorted(missing)}")
    extra = adjustment.keys() - required
    if extra:
        raise GuardrailError(f"note {expected_note_index}: {directive_type} has unexpected keys {sorted(extra)}")

    _check_hours(adjustment["hours"])

    if directive_type == "solar_reduction":
        factor = adjustment["factor"]
        if not isinstance(factor, (int, float)) or isinstance(factor, bool) or not (0 <= factor <= 1):
            raise GuardrailError(f"note {expected_note_index}: factor must be a number in [0, 1]")

    if directive_type == "minimum_battery_reserve":
        val = adjustment["minimum_energy_kwh"]
        if not _finite_nonnegative(val):
            raise GuardrailError(f"note {expected_note_index}: minimum_energy_kwh must be finite and non-negative")
        if val > battery_capacity:
            raise GuardrailError(f"note {expected_note_index}: minimum_energy_kwh exceeds battery capacity")

    if directive_type == "max_grid_window":
        val = adjustment["max_grid_kwh"]
        if not _finite_nonnegative(val):
            raise GuardrailError(f"note {expected_note_index}: max_grid_kwh must be finite and non-negative")

    return DirectiveInterpretation(
        note_index=note_index,
        applies=True,
        directive_type=directive_type,
        structured_adjustment=adjustment,
        explanation=explanation,
    )


def _finite_nonnegative(value):
    try:
        return type(value) in (int, float) and math.isfinite(value) and value >= 0
    except OverflowError:
        return False


def validate_all(raw_entries: List[Dict[str, Any]], num_notes: int, battery_capacity: float) -> List[DirectiveInterpretation]:
    if not isinstance(raw_entries, list):
        raise GuardrailError("top-level output must be a JSON list")
    if len(raw_entries) != num_notes:
        raise GuardrailError(f"expected exactly {num_notes} entries, got {len(raw_entries)}")

    validated = []
    seen_indices = set()
    for i, entry in enumerate(raw_entries):
        parsed = validate_entry(entry, i, battery_capacity)
        if parsed.note_index in seen_indices:
            raise GuardrailError(f"duplicate note_index {parsed.note_index}")
        seen_indices.add(parsed.note_index)
        validated.append(parsed)

    if seen_indices != set(range(num_notes)):
        raise GuardrailError("note_index values must cover 0..N-1 exactly once")

    return validated
