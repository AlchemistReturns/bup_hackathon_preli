"""Run temporary/*.json sample cases against a running GridWise instance.

Usage:
    .venv\\Scripts\\python scripts\\test_sample_cases.py [path/to/cases.json] [--url http://localhost:8000]
"""
import argparse
import json
import sys
import time
from pathlib import Path

import requests

TOL = 0.01


def close(a, b, tol=TOL):
    return abs(a - b) <= tol


def check_case(case, base_url):
    scenario_id = case["id"]
    start = time.perf_counter()
    resp = requests.post(f"{base_url}/optimize-energy", json=case["input"], timeout=60)
    latency_ms = (time.perf_counter() - start) * 1000
    if resp.status_code != 200:
        return False, [f"HTTP {resp.status_code}: {resp.text[:300]}"], latency_ms

    got = resp.json()
    expected = case["expected_output"]
    problems = []

    exp_dirs = expected["directive_interpretation"]
    got_dirs = got.get("directive_interpretation", [])
    if len(got_dirs) != len(exp_dirs):
        problems.append(f"directive_interpretation count {len(got_dirs)} != {len(exp_dirs)}")
    else:
        for e, g in zip(exp_dirs, got_dirs):
            if g.get("applies") != e["applies"]:
                problems.append(f"note {e['note_index']}: applies {g.get('applies')} != {e['applies']}")
            if e["applies"] and g.get("directive_type") != e["directive_type"]:
                problems.append(
                    f"note {e['note_index']}: directive_type {g.get('directive_type')} != {e['directive_type']}"
                )

    for field in ("total_grid_kwh", "total_cost_bdt", "peak_grid_kwh"):
        if not close(got.get(field, float("nan")), expected[field]):
            problems.append(f"{field}: got {got.get(field)} != expected {expected[field]}")

    if len(got.get("hourly_plan", [])) != 24:
        problems.append(f"hourly_plan has {len(got.get('hourly_plan', []))} entries, expected 24")

    return (len(problems) == 0), problems, latency_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "cases_file",
        nargs="?",
        default="temporary/BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json",
    )
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()

    data = json.loads(Path(args.cases_file).read_text(encoding="utf-8"))
    cases = data["cases"]

    passed = 0
    latencies = []
    for case in cases:
        ok, problems, latency_ms = check_case(case, args.url)
        latencies.append(latency_ms)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {case['id']} — {case.get('label', '')} — {latency_ms:.1f} ms")
        for p in problems:
            print(f"    - {p}")
        passed += ok

    latencies.sort()
    n = len(latencies)
    median = latencies[n // 2] if n % 2 else (latencies[n // 2 - 1] + latencies[n // 2]) / 2
    p95 = latencies[min(int(round(0.95 * (n - 1))), n - 1)]
    print(f"\n{passed}/{len(cases)} passed; median={median:.1f} ms; p95={p95:.1f} ms")
    sys.exit(0 if passed == len(cases) else 1)


if __name__ == "__main__":
    main()
