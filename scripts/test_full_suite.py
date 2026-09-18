import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient
from app.main import app

TOL = 0.01

def close(a, b, tol=TOL):
    return abs(a - b) <= tol

DEFAULT_CASES_FILE = Path(__file__).resolve().parent.parent / "temporary" / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"

def main():
    client = TestClient(app)
    cases_file = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CASES_FILE
    data = json.loads(cases_file.read_text(encoding="utf-8"))
    cases = data["cases"]

    passed = 0
    for case in cases:
        scenario_id = case["id"]
        label = case.get("label", "")
        resp = client.post("/optimize-energy", json=case["input"])
        if resp.status_code != 200:
            print(f"[FAIL] {scenario_id} - HTTP {resp.status_code}: {resp.text[:200]}")
            continue

        got = resp.json()
        expected = case["expected_output"]
        problems = []

        exp_dirs = expected["directive_interpretation"]
        got_dirs = got.get("directive_interpretation", [])
        if len(got_dirs) != len(exp_dirs):
            problems.append(f"directive count {len(got_dirs)} != {len(exp_dirs)}")
        else:
            for e, g in zip(exp_dirs, got_dirs):
                if g.get("applies") != e["applies"]:
                    problems.append(f"note {e['note_index']}: applies {g.get('applies')} != {e['applies']}")
                if e["applies"] and g.get("directive_type") != e["directive_type"]:
                    problems.append(f"note {e['note_index']}: type {g.get('directive_type')} != {e['directive_type']}")

        for field in ("total_grid_kwh", "total_cost_bdt", "peak_grid_kwh"):
            g_val = got.get(field, float("nan"))
            e_val = expected[field]
            if not close(g_val, e_val):
                problems.append(f"{field}: got {g_val} != expected {e_val}")

        if len(got.get("hourly_plan", [])) != 24:
            problems.append(f"hourly_plan has {len(got.get('hourly_plan', []))} entries, expected 24")

        ok = len(problems) == 0
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {scenario_id} — {label}")
        for p in problems:
            print(f"    - {p}")
        passed += ok

    print(f"\n{passed}/{len(cases)} passed")
    return passed == len(cases)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
