"""
Deterministic validation of the LLM's raw directive output.
The LLM's JSON is never trusted directly — every field is checked here
before anything is handed to the optimizer. Any failure raises
GuardrailError with a message that is fed back to the LLM for a retry.
"""
import re
from typing import Any, Dict, List, Optional
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


def _parse_time(t_str: str) -> Optional[int]:
    t_str = t_str.strip().lower()
    if t_str == "noon":
        return 12
    if t_str == "midnight":
        return 0
    m = re.match(r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)", t_str)
    if not m:
        return None
    hr = int(m.group(1))
    period = m.group(3)
    if period == "am":
        return 0 if hr == 12 else hr
    else:
        return 12 if hr == 12 else hr + 12


def extract_time_window(text: str) -> Optional[List[int]]:
    """Extract 0-indexed hour window from natural language like 'from 6 PM until 9 PM'."""
    m = re.search(
        r"(?:from|between)\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)|noon|midnight)\s+(?:until|to|and)\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)|noon|midnight)",
        text,
        re.I,
    )
    if not m:
        return None
    start = _parse_time(m.group(1))
    end = _parse_time(m.group(2))
    if start is not None and end is not None and end > start:
        return list(range(start, end))
    return None


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


def validate_entry(
    entry: Dict[str, Any],
    expected_note_index: int,
    battery_capacity: float,
    note_text: Optional[str] = None,
) -> DirectiveInterpretation:
    if not isinstance(entry, dict):
        raise GuardrailError(f"entry for note {expected_note_index} is not an object")

    note_index = entry.get("note_index")
    if note_index != expected_note_index:
        raise GuardrailError(
            f"expected note_index {expected_note_index} at this position, got {note_index!r}"
        )

    directive_type = entry.get("directive_type")
    if directive_type not in DIRECTIVE_TYPES:
        raise GuardrailError(
            f"note {expected_note_index}: directive_type must be one of {sorted(DIRECTIVE_TYPES)}, got {directive_type!r}"
        )

    applies = entry.get("applies")
    adjustment = entry.get("structured_adjustment")
    explanation = entry.get("explanation", "") or ""

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

    # If note_text has an unambiguous time window, ensure adjustment['hours'] matches
    if note_text and "hours" in adjustment:
        det_window = extract_time_window(note_text)
        if det_window is not None:
            adjustment["hours"] = det_window

    _check_hours(adjustment["hours"])

    if directive_type == "solar_reduction":
        factor = adjustment["factor"]
        if not isinstance(factor, (int, float)) or isinstance(factor, bool) or not (0 <= factor <= 1):
            raise GuardrailError(f"note {expected_note_index}: factor must be a number in [0, 1]")

    if directive_type == "minimum_battery_reserve":
        # Check if note expressed reserve as a percentage of battery capacity
        if note_text:
            pct_m = re.search(r"(\d+(?:\.\d+)?)\s*%\s*(?:of\s+(?:the\s+)?battery)?", note_text, re.I)
            if pct_m:
                calc_kwh = (float(pct_m.group(1)) / 100.0) * battery_capacity
                adjustment["minimum_energy_kwh"] = round(calc_kwh, 4)

        val = adjustment["minimum_energy_kwh"]
        if not isinstance(val, (int, float)) or isinstance(val, bool) or val < 0:
            raise GuardrailError(f"note {expected_note_index}: minimum_energy_kwh must be a non-negative number")
        if val > battery_capacity:
            raise GuardrailError(f"note {expected_note_index}: minimum_energy_kwh exceeds battery capacity")

    if directive_type == "max_grid_window":
        val = adjustment["max_grid_kwh"]
        if not isinstance(val, (int, float)) or isinstance(val, bool) or val < 0:
            raise GuardrailError(f"note {expected_note_index}: max_grid_kwh must be a non-negative number")

    return DirectiveInterpretation(
        note_index=note_index,
        applies=True,
        directive_type=directive_type,
        structured_adjustment=adjustment,
        explanation=explanation,
    )


def validate_all(
    raw_entries: List[Dict[str, Any]],
    num_notes: int,
    battery_capacity: float,
    operator_notes: Optional[List[str]] = None,
) -> List[DirectiveInterpretation]:
    if not isinstance(raw_entries, list):
        raise GuardrailError("top-level output must be a JSON list")
    if len(raw_entries) != num_notes:
        raise GuardrailError(f"expected exactly {num_notes} entries, got {len(raw_entries)}")

    validated = []
    seen_indices = set()
    for i, entry in enumerate(raw_entries):
        note_text = operator_notes[i] if operator_notes and i < len(operator_notes) else None
        parsed = validate_entry(entry, i, battery_capacity, note_text=note_text)
        if parsed.note_index in seen_indices:
            raise GuardrailError(f"duplicate note_index {parsed.note_index}")
        seen_indices.add(parsed.note_index)
        validated.append(parsed)

    if seen_indices != set(range(num_notes)):
        raise GuardrailError("note_index values must cover 0..N-1 exactly once")

    return validated
