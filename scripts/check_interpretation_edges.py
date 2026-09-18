"""Paid, uncached LLM checks with novel wording, values and clock boundaries."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.graph import _run_uncached

# Test data only; production code never imports this module.
CASES = [
    ("percent_reserve", 320, "From 18:00 to 21:00 today, retain no less than 17.5 percent of the battery's rated capacity.", "minimum_battery_reserve", {"hours": [18, 19, 20], "minimum_energy_kwh": 56}),
    ("solar_reduced_by", 200, "PV generation will be reduced by 65% between 10 AM and noon today.", "solar_reduction", {"hours": [10, 11], "factor": 0.35}),
    ("solar_remaining", 200, "Only 65% of predicted solar is usable from noon to 3 PM today.", "solar_reduction", {"hours": [12, 13, 14], "factor": 0.65}),
    ("overnight", 200, "Keep the charging circuit disconnected from 11 PM until 2 AM.", "no_charge_window", {"hours": [0, 1, 23]}),
    ("midnight_end", 200, "Do not draw energy out of the battery from 9 PM until midnight.", "no_discharge_window", {"hours": [21, 22, 23]}),
    ("midnight_start", 200, "Battery charging is prohibited from midnight until 3 AM.", "no_charge_window", {"hours": [0, 1, 2]}),
    ("single_hour", 200, "At 8 PM, restrict grid purchases to 172.5 kWh for that hour.", "max_grid_window", {"hours": [20], "max_grid_kwh": 172.5}),
    ("disjoint", 200, "Charging must be blocked from 1 AM until 3 AM and again from 7 AM until 9 AM.", "no_charge_window", {"hours": [1, 2, 7, 8]}),
    ("whole_day", 200, "Battery discharge is disabled for all of today.", "no_discharge_window", {"hours": list(range(24))}),
    ("future_energy", 200, "Next week, the battery charger will be serviced from 2 PM to 4 PM; today's operation is unchanged.", "no_op", None),
    ("unrelated", 200, "The alumni office has rescheduled tomorrow's awards ceremony.", "no_op", None),
    ("absolute_reserve", 200, "For the interval 17:00 to 20:00, battery energy must never fall below 73.25 kWh.", "minimum_battery_reserve", {"hours": [17, 18, 19], "minimum_energy_kwh": 73.25}),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    records = []
    for name, capacity, note, kind, adjustment in CASES:
        started = perf_counter()
        record = {"case_id": name, "passed": False}
        try:
            result = _run_uncached([note], capacity)[0]
            record["passed"] = result.directive_type == kind and result.structured_adjustment == adjustment
            if not record["passed"]:
                record["returned"] = result.model_dump()
        except Exception as exc:
            record["error"] = type(exc).__name__
        record["latency_ms"] = round((perf_counter() - started) * 1000, 3)
        records.append(record)
        print(f"{name}: {'PASS' if record['passed'] else 'FAIL'} {record['latency_ms']:.1f} ms", flush=True)
    report = {"passed": sum(r["passed"] for r in records), "total": len(records), "cases": records}
    if args.report:
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"{report['passed']}/{report['total']} passed")
    return 0 if report["passed"] == report["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
