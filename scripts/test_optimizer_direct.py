import json
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.schemas import BatteryConfig, HourEntry, DirectiveInterpretation
from app.optimizer import solve
from app.replay import replay

def test_all():
    with open(r"d:\buppreli\BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    cases = data["cases"]
    all_ok = True
    for case in cases:
        inp = case["input"]
        exp = case["expected_output"]
        battery = BatteryConfig(**inp["battery"])
        hours = [HourEntry(**h) for h in inp["hours"]]
        directives = [DirectiveInterpretation(**d) for d in exp["directive_interpretation"]]

        plan = solve(hours, battery, directives)
        tot_grid, tot_cost, peak_grid = replay(hours, battery, directives, plan)

        grid_diff = abs(tot_grid - exp["total_grid_kwh"])
        cost_diff = abs(tot_cost - exp["total_cost_bdt"])
        peak_diff = abs(peak_grid - exp["peak_grid_kwh"])

        ok = grid_diff <= 0.05 and cost_diff <= 0.05 and peak_diff <= 0.05
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {case['id']}:")
        print(f"  Grid: got {tot_grid:.2f}, exp {exp['total_grid_kwh']:.2f} (diff: {grid_diff:.4f})")
        print(f"  Cost: got {tot_cost:.2f}, exp {exp['total_cost_bdt']:.2f} (diff: {cost_diff:.4f})")
        print(f"  Peak: got {peak_grid:.2f}, exp {exp['peak_grid_kwh']:.2f} (diff: {peak_diff:.4f})")

        if not ok:
            all_ok = False

    print("\nSummary:", "ALL TESTS PASSED!" if all_ok else "SOME TESTS FAILED!")
    return all_ok

if __name__ == "__main__":
    test_all()
