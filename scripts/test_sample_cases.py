"""Run temporary/*.json sample cases against a running GridWise instance.

Usage:
    .venv\\Scripts\\python scripts\\test_sample_cases.py [path/to/cases.json] [--url http://localhost:8000]
"""
import argparse
import json
import sys
from pathlib import Path

import requests

TOL = 0.01


def close(a, b, tol=TOL):
    return abs(a - b) <= tol


def check_case(case, base_url):
    scenario_id = case["id"]
    resp = requests.post(f"{base_url}/optimize-energy", json=case["input"], timeout=60)
    if resp.status_code != 200:
        return False, [f"HTTP {resp.status_code}: {resp.text[:300]}"]

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

    return (len(problems) == 0), problems


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
    for case in cases:
        ok, problems = check_case(case, args.url)
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {case['id']} — {case.get('label', '')}")
        for p in problems:
            print(f"    - {p}")
        passed += ok

    print(f"\n{passed}/{len(cases)} passed")
    sys.exit(0 if passed == len(cases) else 1)


if __name__ == "__main__":
    main()
