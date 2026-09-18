"""Convert model-extracted clock boundaries into the public hours schema.

The model still interprets language and selects the directive, times and values.
Code owns interval expansion; there are no phrase or public-case lookups here.
"""
from app.guardrails import GuardrailError, REQUIRED_KEYS


def normalize_time_windows(raw_entries):
    if not isinstance(raw_entries, list):
        raise GuardrailError("Expected a directives list")
    normalized = []
    for index, entry in enumerate(raw_entries):
        if not isinstance(entry, dict):
            raise GuardrailError(f"note {index}: expected an object")
        kind = entry.get("directive_type")
        adjustment = entry.get("structured_adjustment")
        if kind == "no_op":
            if adjustment is not None:
                raise GuardrailError(f"note {index}: no_op adjustment must be null")
            normalized.append(dict(entry))
            continue
        if not isinstance(kind, str) or kind not in REQUIRED_KEYS:
            raise GuardrailError(f"note {index}: unsupported directive type")
        if not isinstance(adjustment, dict):
            raise GuardrailError(f"note {index}: expected an adjustment object")
        required = (REQUIRED_KEYS[kind] - {"hours"}) | {"time_windows"}
        if adjustment.keys() != required:
            raise GuardrailError(f"note {index}: adjustment requires time_windows and the directive value")
        windows = adjustment["time_windows"]
        if not isinstance(windows, list) or not 1 <= len(windows) <= 24:
            raise GuardrailError(f"note {index}: expected 1-24 time windows")
        hours = set()
        for window in windows:
            if not isinstance(window, dict) or window.keys() != {"start_hour", "end_hour"}:
                raise GuardrailError(f"note {index}: each window requires start_hour and end_hour")
            start, end = window["start_hour"], window["end_hour"]
            if type(start) is not int or type(end) is not int:
                raise GuardrailError(f"note {index}: time boundaries must be integers")
            if not 0 <= start <= 23 or not 1 <= end <= 24 or start == end:
                raise GuardrailError(f"note {index}: invalid boundaries; use 0 to 24 for a full day")
            if start < end:
                hours.update(range(start, end))
            else:
                # A window crossing midnight: e.g. 23 to 2 -> [0, 1, 23].
                hours.update(range(start, 24))
                hours.update(range(0, end))
        converted = {key: value for key, value in adjustment.items() if key != "time_windows"}
        converted["hours"] = sorted(hours)
        normalized.append({**entry, "structured_adjustment": converted})
    return normalized
