"""Expand the compact model result into the unchanged public directive schema."""
from app.guardrails import GuardrailError
from app.time_windows import normalize_time_windows

PROTOCOL_VERSION = "numeric-clock-boundaries-v2"
PACKED_TYPES = ("ignore", "solar", "reserve", "no_charge", "no_discharge", "grid_cap")
TYPES = {
    "solar": ("solar_reduction", "factor"),
    "reserve": ("minimum_battery_reserve", "minimum_energy_kwh"),
    "no_charge": ("no_charge_window", None),
    "no_discharge": ("no_discharge_window", None),
    "grid_cap": ("max_grid_window", "max_grid_kwh"),
    "ignore": ("no_op", None),
}


def expand_packed_directives(decoded):
    """Decode [type, value, start, end, ...]; code validates every position."""
    if not isinstance(decoded, dict) or decoded.keys() != {"d"} or not isinstance(decoded["d"], list):
        raise GuardrailError("Expected an object with a d array")
    entries = []
    for index, row in enumerate(decoded["d"]):
        if not isinstance(row, list) or not 2 <= len(row) <= 50 or len(row) % 2:
            raise GuardrailError(f"note {index}: expected [type,value,start,end,...]")
        code, value = row[:2]
        if type(code) is not int or not 0 <= code < len(PACKED_TYPES):
            raise GuardrailError(f"note {index}: type code must be an integer from 0 to 5")
        if type(value) not in (int, float):
            raise GuardrailError(f"note {index}: value must be numeric")
        if code in (0, 3, 4) and value != 0:
            raise GuardrailError(f"note {index}: unused value must be zero")
        entries.append({
            "t": PACKED_TYPES[code],
            "v": None if code in (0, 3, 4) else value,
            "w": [{"s": row[i], "e": row[i + 1]} for i in range(2, len(row), 2)],
        })
    return expand_compact_directives({"d": entries})


def expand_compact_directives(decoded):
    if not isinstance(decoded, dict) or decoded.keys() != {"d"} or not isinstance(decoded["d"], list):
        raise GuardrailError("Expected an object with a d array")
    expanded = []
    for index, item in enumerate(decoded["d"]):
        if not isinstance(item, dict) or item.keys() != {"t", "w", "v"}:
            raise GuardrailError(f"note {index}: expected t, w and v")
        if not isinstance(item["t"], str) or item["t"] not in TYPES:
            raise GuardrailError(f"note {index}: unsupported directive")
        kind, field = TYPES[item["t"]]
        windows, value = item["w"], item["v"]
        if not isinstance(windows, list):
            raise GuardrailError(f"note {index}: w must be an array")
        if field is None and value is not None:
            raise GuardrailError(f"note {index}: v must be null for {item['t']}")
        if field is not None and type(value) not in (int, float):
            raise GuardrailError(f"note {index}: v must be numeric for {item['t']}")
        if kind == "no_op":
            if windows:
                raise GuardrailError(f"note {index}: ignore must have no windows")
            adjustment = None
        else:
            translated = []
            for window in windows:
                if not isinstance(window, dict) or window.keys() != {"s", "e"}:
                    raise GuardrailError(f"note {index}: each window requires s and e")
                translated.append({"start_hour": window["s"], "end_hour": window["e"]})
            adjustment = {"time_windows": translated}
            if field:
                adjustment[field] = value
        expanded.append({
            "note_index": index, "applies": kind != "no_op",
            "directive_type": kind, "structured_adjustment": adjustment,
            "explanation": "Pending deterministic explanation.",
        })
    normalized = normalize_time_windows(expanded)
    for entry in normalized:
        kind, adjustment = entry["directive_type"], entry["structured_adjustment"]
        if kind == "no_op":
            entry["explanation"] = "This note does not affect the current energy schedule."
            continue
        descriptions = {
            "solar_reduction": "Use the specified remaining fraction of forecast solar",
            "minimum_battery_reserve": "Maintain the specified minimum battery reserve",
            "no_charge_window": "Battery charging is unavailable",
            "no_discharge_window": "Battery discharging is unavailable",
            "max_grid_window": "Limit grid imports to the specified maximum",
        }
        numbers = {key: value for key, value in adjustment.items() if key != "hours"}
        suffix = "; " + ", ".join(f"{key}={value}" for key, value in numbers.items()) if numbers else ""
        entry["explanation"] = (
            descriptions[kind] + " during hours " + ", ".join(map(str, adjustment["hours"])) + suffix + "."
        )
    return normalized
