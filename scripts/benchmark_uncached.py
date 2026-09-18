"""Alternate old/new servers on identical cases; reject answer-cache hits."""
import argparse
import json
import math
from pathlib import Path
import statistics

from check_samples import run_cases


def summarize(records):
    values = sorted(row["latency_ms"] for row in records)
    return {"count": len(records), "passed": sum(row["passed"] for row in records),
            "uncached": all(row.get("interpretation_cache") == "disabled" for row in records),
            "median_ms": statistics.median(values),
            "p95_ms": values[math.ceil(0.95 * len(values)) - 1], "max_ms": max(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-url", required=True)
    parser.add_argument("--candidate-url", required=True)
    parser.add_argument("--repeat", type=int, default=2)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.repeat < 1:
        parser.error("--repeat must be positive")
    cases = json.loads((Path(__file__).resolve().parents[1] / "tests/fixtures/public_cases.json").read_text(encoding="utf-8"))["cases"]
    records = {"baseline": [], "candidate": []}
    urls = {"baseline": args.baseline_url, "candidate": args.candidate_url}
    for run in range(args.repeat):
        for index, case in enumerate(cases):
            # Alternate order to reduce drift from short-term network/provider load.
            order = ("baseline", "candidate") if (run + index) % 2 == 0 else ("candidate", "baseline")
            for name in order:
                print(f"{name} pass={run+1}", flush=True)
                row = run_cases([case], urls[name])[0]
                row["run"] = run + 1
                records[name].append(row)
    summary = {name: summarize(rows) for name, rows in records.items()}
    print(json.dumps(summary, indent=2))
    args.report.write_text(json.dumps({"summary": summary, "cases": records}, indent=2, allow_nan=False), encoding="utf-8")
    return 0 if all(s["uncached"] and s["passed"] == s["count"] for s in summary.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
