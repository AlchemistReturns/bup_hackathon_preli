"""Optimizer-only harness for the public cases (bypasses the LLM).

Feeds the reference directive_interpretation straight into optimizer.solve(), replays the plan
with app.replay, and compares total_cost_bdt to the reference. On failure it dumps the raw solver
values of charge/discharge (old split model) or bat (signed model) for every hour.

Usage:  .venv\\Scripts\\python scripts\\baseline_public_cases.py [cases.json]
"""
import json
import sys
from pathlib import Path

import pulp

from app.optimizer import solve
from app.replay import ReplayError, replay
from app.schemas import DirectiveInterpretation, ScenarioRequest

DEFAULT = Path(__file__).resolve().parent.parent / "temporary" / "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"

_last = {}
_orig_solve = pulp.LpProblem.solve


def _capturing_solve(self, *a, **kw):
    r = _orig_solve(self, *a, **kw)
    _last["problem"] = self
    return r


pulp.LpProblem.solve = _capturing_solve


def raw_values(prob):
    return {v.name: v.value() for v in prob.variables()}


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    cases = json.loads(path.read_text(encoding="utf-8"))["cases"]
    print(f"pulp {pulp.__version__}")
    rows, details = [], []
    for case in cases:
        req = ScenarioRequest(**case["input"])
        ref = case["expected_output"]
        dirs = [DirectiveInterpretation(**d) for d in ref["directive_interpretation"]]
        _last.clear()
        err, cost = None, None
        try:
            out = solve(req.hours, req.battery, dirs)
            plan = out[0] if isinstance(out, tuple) else out
            _, cost, _ = replay(req.hours, req.battery, dirs, plan)
        except ReplayError as e:
            err = str(e)
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        rows.append((case["id"], ref["total_cost_bdt"], cost, err))

        vals = raw_values(_last["problem"]) if "problem" in _last else {}
        simul = [
            (h, vals.get(f"charge_{h}"), vals.get(f"discharge_{h}"))
            for h in range(24)
            if (vals.get(f"charge_{h}") or 0) > 1e-6 and (vals.get(f"discharge_{h}") or 0) > 1e-6
        ]
        details.append((case["id"], err, simul, vals))

    print(f"\n{'case':<11}{'ref':>10}{'ours':>10}{'delta':>8}  replay")
    for cid, ref, cost, err in rows:
        if cost is None:
            print(f"{cid:<11}{ref:>10.2f}{'-':>10}{'-':>8}  RAISED: {err}")
        else:
            print(f"{cid:<11}{ref:>10.2f}{cost:>10.2f}{cost - ref:>8.2f}  ok")

    print("\n--- raw solver diagnostics (old split model only) ---")
    for cid, err, simul, vals in details:
        has_split = any(k.startswith("charge_") for k in vals)
        if not has_split:
            continue
        print(f"{cid}: hours with BOTH charge>0 and discharge>0 in raw solver output: {len(simul)}"
              f"{'  (replay raised)' if err else ''}")
        for h, c, d in simul:
            print(f"    hour {h:2d}: charge={c:.4f} discharge={d:.4f} net={c - d:+.4f}")
        if err:
            print(f"    replay error: {err}")


if __name__ == "__main__":
    main()
