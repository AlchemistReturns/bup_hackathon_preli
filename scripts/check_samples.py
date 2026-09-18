"""Run every public case, checking interpretation, ground-truth physics and cost."""
import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from time import perf_counter
from urllib.error import HTTPError
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.guardrails import validate_all
from app.optimizer import solve
from app.replay import replay, replay_response
from app.schemas import OptimizeEnergyResponse, ScenarioRequest


def same_interpretation(actual, expected):
    if len(actual) != len(expected):
        return False
    for got, want in zip(actual, expected):
        for field in ("note_index", "applies", "directive_type"):
            if got[field] != want[field]:
                return False
        left, right = got["structured_adjustment"], want["structured_adjustment"]
        if right is None:
            if left is not None:
                return False
            continue
        if not isinstance(left, dict) or left.keys() != right.keys():
            return False
        for key in right:
            if key == "hours":
                if left[key] != right[key]:
                    return False
            elif not math.isfinite(left[key]) or abs(left[key] - right[key]) > 0.01:
                return False
    return True


def evaluate_response(case, result):
    """Public evaluator uses organizer directives, never trusts reported ones."""
    request = ScenarioRequest.model_validate(case["input"])
    expected = case["expected_output"]
    truth = validate_all(expected["directive_interpretation"], len(request.operator_notes), request.battery.capacity_kwh)
    flags = {
        "interpretation": same_interpretation(
            [item.model_dump() for item in result.directive_interpretation],
            expected["directive_interpretation"],
        ),
        "reported_plan_valid": False,
        "ground_truth_valid": False,
        "optimal_cost": False,
    }
    try:
        replay_response(request, result)
        flags["reported_plan_valid"] = True
    except Exception as exc:
        flags["reported_plan_error"] = type(exc).__name__
    try:
        _, cost, _ = replay(request.hours, request.battery, truth, result.hourly_plan)
        flags["ground_truth_valid"] = True
        flags["optimal_cost"] = abs(cost - expected["total_cost_bdt"]) <= 0.01
    except Exception as exc:
        flags["ground_truth_error"] = type(exc).__name__
    flags["passed"] = all(flags[key] for key in (
        "interpretation", "reported_plan_valid", "ground_truth_valid", "optimal_cost",
    ))
    flags["cost_bdt"] = result.total_cost_bdt
    flags["expected_cost_bdt"] = expected["total_cost_bdt"]
    return flags


def run_cases(cases, base_url=None, repeat=1):
    records = []
    mode = "LIVE API: language + optimizer" if base_url else "OFFLINE: published directives (no LLM)"
    print(mode, flush=True)
    for run in range(1, repeat + 1):
        for case in cases:
            started = perf_counter()
            record = {"case_id": case["id"], "run": run, "passed": False}
            try:
                request = ScenarioRequest.model_validate(case["input"])
                expected = case["expected_output"]
                if base_url:
                    call = Request(
                        base_url.rstrip("/") + "/optimize-energy",
                        data=request.model_dump_json().encode(),
                        headers={"Content-Type": "application/json"}, method="POST",
                    )
                    with urlopen(call, timeout=30) as response:
                        record["http_status"] = response.status
                        headers = getattr(response, "headers", {})
                        record["interpretation_cache"] = headers.get("X-Interpretation-Cache", "unknown")
                        record["server_timing"] = headers.get("Server-Timing", "")
                        result = OptimizeEnergyResponse.model_validate_json(response.read())
                    record.update(evaluate_response(case, result))
                    if not record["passed"]:
                        record["returned_directives"] = [item.model_dump() for item in result.directive_interpretation]
                else:
                    truth = validate_all(expected["directive_interpretation"], len(request.operator_notes), request.battery.capacity_kwh)
                    plan = solve(request.hours, request.battery, truth)
                    _, cost, _ = replay(request.hours, request.battery, truth, plan)
                    record.update(
                        passed=abs(cost - expected["total_cost_bdt"]) <= 0.01,
                        ground_truth_valid=True,
                        cost_bdt=cost,
                        expected_cost_bdt=expected["total_cost_bdt"],
                    )
            except HTTPError as exc:
                record["http_status"] = exc.code
                record["error"] = "HTTPError"
            except Exception as exc:
                # Never print potentially sensitive provider/body exception text.
                record["error"] = type(exc).__name__
            record["latency_ms"] = round((perf_counter() - started) * 1000, 3)
            records.append(record)
            status = "PASS" if record["passed"] else "FAIL"
            details = f"cost={record['cost_bdt']:.6f} BDT" if "cost_bdt" in record else record.get("error", "")
            if base_url and "interpretation" in record:
                details += (
                    f" interpretation={record['interpretation']}"
                    f" ground_truth_valid={record['ground_truth_valid']}"
                    f" optimal_cost={record['optimal_cost']}"
                    f" cache={record.get('interpretation_cache', 'unknown')}"
                )
            print(f"run={run} {case['id']}: {status} {details} latency={record['latency_ms']:.1f} ms", flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", help="Call a live API (may incur model charges)")
    parser.add_argument("--repeat", type=int, default=1, help="Runs of all cases; repeated API requests may hit the server cache")
    parser.add_argument("--report", type=Path, help="Save per-case metrics to this JSON file")
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    path = Path(__file__).resolve().parents[1] / "tests/fixtures/public_cases.json"
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    records = run_cases(cases, args.base_url, args.repeat)
    passed = sum(record["passed"] for record in records)
    ordered = sorted(record["latency_ms"] for record in records)
    p95 = ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]
    print(f"{passed}/{len(records)} passed; median={statistics.median(ordered):.1f} ms; p95={p95:.1f} ms")
    if args.base_url:
        for source in sorted({record.get("interpretation_cache", "unknown") for record in records}):
            latencies = sorted(record["latency_ms"] for record in records
                               if record.get("interpretation_cache", "unknown") == source)
            tail = latencies[max(0, math.ceil(0.95 * len(latencies)) - 1)]
            print(f"  cache={source}: n={len(latencies)}; median={statistics.median(latencies):.1f} ms; p95={tail:.1f} ms")
    if args.report:
        args.report.write_text(json.dumps({"passed": passed, "total": len(records), "cases": records}, indent=2, allow_nan=False), encoding="utf-8")
    return 0 if passed == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
