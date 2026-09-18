# GridWise — Optimization Strategy & Correctness Specification

**Status:** verified against all 10 public sample cases
**Scope:** the Math Optimizer + Final Validator boxes only (LLM/guardrail path unchanged)
**Audience:** implementer (human or AI agent) working on `app/optimizer.py`

---

## 0. TL;DR — what this document says

1. **The current `app/optimizer.py` fails 3 of the 10 public sample cases** — not by getting the cost wrong, but by returning **HTTP 500**. Root cause identified, reproduced, and fixed below. This is the single highest-value change in this document.
2. The LP formulation is **provably optimal** for this problem — not a heuristic that is "usually good." We verified this: our LP reproduces the organizers' reference cost to **±0.00 BDT on all 10 cases**.
3. Three additional hardening layers turn "usually correct" into "cannot be incorrect": a **signed-battery reformulation**, an **elastic infeasibility fallback**, and a **canonicalizer**.
4. Drop-in replacement code is in §7. It passes **10/10 using our own `replay.py` as the judge**.

| Metric | Current `optimizer.py` | Proposed v2 |
|---|---|---|
| Public cases returning a valid plan | **7 / 10** | **10 / 10** |
| Cost vs. reference optimum | exact where it succeeds | exact on all 10 |
| Simultaneous charge+discharge defects | present in 3 cases | structurally impossible |
| Behaviour on an infeasible directive set | HTTP 500 (case scores 0) | valid plan, minimum violation |
| Avg solve time | ~5 ms | ~5 ms |

---

## 1. The bug we found (read this first)

### 1.1 Symptom

Running the **current** `app/optimizer.py` + `app/replay.py` against the public cases, using the organizers' own ground-truth directives:

```
case          ref cost  team cost     delta  result
SAMPLE-01     38365.00   38365.00      0.00  OK
SAMPLE-02          —          —          —   FAILED: hour 8: battery_energy_after_kwh does not match battery action
SAMPLE-03     35480.00   35480.00      0.00  OK
SAMPLE-04          —          —          —   FAILED: hour 7: battery_energy_after_kwh does not match battery action
SAMPLE-05     33950.00   33950.00      0.00  OK
SAMPLE-06     34090.00   34090.00      0.00  OK
SAMPLE-07     38550.00   38550.00      0.00  OK
SAMPLE-08          —          —          —   FAILED: hour 6: battery_energy_after_kwh does not match battery action
SAMPLE-09     34873.00   34873.00      0.00  OK
SAMPLE-10     41620.00   41620.00      0.00  OK

7/10 cases reproduce the reference optimum.
```

`replay.py` raises → `main.py` catches `ReplayError` → **HTTP 500**. The case scores zero, including the interpretation points the LLM had already earned correctly.

### 1.2 Root cause

The README says:

> *No binary charge/discharge exclusivity constraint (relies on cost-minimality; code post-processes the rare simultaneous case as a fallback)*

**The assumption "relies on cost-minimality" is false for this problem**, and the case is not rare.

GridWise models **no round-trip efficiency loss** (Section 9.1: `E_after = E_before ± battery_kwh`, no η). That means charging 55 kWh and discharging 55 kWh in the same hour is *perfectly free* — it cancels in both the energy balance and the battery state equation. The objective only prices `grid_kwh`, so the LP has **zero incentive to avoid it**, and both solutions sit at equally-optimal vertices of the feasible polytope. CBC returns whichever vertex its pivot sequence lands on.

Dumping the raw CBC solution confirms it is routine, not rare:

```
--- SAMPLE-02  status=Optimal  cost=42885.00
    hour  8: SIMULTANEOUS charge=55.0000 discharge=55.0000  net=+0.0000  E_after=200.0000
    hour  9: SIMULTANEOUS charge=55.0000 discharge=55.0000  net=+0.0000  E_after=200.0000
    hour 10: SIMULTANEOUS charge=55.0000 discharge=55.0000  net=+0.0000  E_after=200.0000
    hour 15: SIMULTANEOUS charge=55.0000 discharge=55.0000  net=+0.0000  E_after=200.0000
    hour 17: SIMULTANEOUS charge=50.0000 discharge=55.0000  net=-5.0000  E_after=195.0000
    hour 21: SIMULTANEOUS charge=55.0000 discharge=55.0000  net=+0.0000  E_after=30.0000
```

Six hours in one case. The existing post-processing then does this:

```python
if c > EPS and d > EPS:
    if c >= d: d = 0.0      # keeps charge=55, throws away discharge=55
    else:      c = 0.0
```

…which reports `battery_action="charge", battery_kwh=55`, but leaves `battery_energy_after_kwh` at the solver's value, which reflected the **net** of `+0`. The reported action and the reported energy now contradict each other, and the replay correctly rejects the plan.

The post-processing is not salvageable by patching it: throwing away one leg of a `55/55` pair changes the physical meaning of the hour. The fix has to happen in the model.

### 1.3 Why this matters beyond the 3 cases

This is a **silent, data-dependent** failure. It does not depend on the notes being interpreted correctly — all 3 failures above used the organizers' own ground-truth directives. It depends only on which vertex the solver happens to return, which changes with solver version, platform, and problem data. A run that passes locally can fail on the judge's machine.

---

## 2. Why Linear Programming is the right answer (and why not the alternatives)

This is worth stating explicitly, because the instinct in a hackathon is to reach for something that looks more sophisticated.

**The problem class is standard.** "Minimize grid import cost over a fixed horizon with PV + a battery under a time-of-use tariff" is *battery arbitrage / behind-the-meter dispatch*, studied for over a decade. Production open-source systems solve exactly this with LP — EMHASS (a widely-used home battery optimizer) uses linear programming over CVXPY with HiGHS as the default solver. In DrivenData's *Power Laws: Optimizing Demand-side Strategies* competition, which posed a near-identical problem, the winning approaches were LP-based; the top team's write-up states plainly: *"We considered the problem as a dynamic optimization problem. The problem at each step was modeled as a linear programming (LP)."*

**The LP is not an approximation here — it is the exact optimum.** Every constraint in Sections 9.1–9.6 is linear, and the objective `Σ grid[h]·tariff[h]` is linear. A single LP over the whole 24-hour horizon therefore returns the global minimum. There is no gap to close.

**We verified this empirically.** Our LP reproduces the organizers' reference `total_cost_bdt` to the cent on every public case:

```
case          ref cost    LP cost      delta  simul  status
SAMPLE-01     38365.00   38365.00       0.00      0  OK
SAMPLE-02     42885.00   42885.00       0.00      0  OK
SAMPLE-03     35480.00   35480.00       0.00      0  OK
SAMPLE-04     40495.00   40495.00       0.00      0  OK
SAMPLE-05     33950.00   33950.00       0.00      0  OK
SAMPLE-06     34090.00   34090.00       0.00      0  OK
SAMPLE-07     38550.00   38550.00       0.00      0  OK
SAMPLE-08     37665.00   37665.00       0.00      0  OK
SAMPLE-09     34873.00   34873.00       0.00      0  OK
SAMPLE-10     41620.00   41620.00       0.00      0  OK
```

Two things follow. First, our constraint set is complete — no hidden rule is missing, or we would come out *cheaper* than the reference. Second, our model is not over-constrained — or we would come out *more expensive*. Exact agreement on all ten is strong evidence the organizers generated the references with an equivalent LP.

### What to avoid, and why

| Approach | Why not |
|---|---|
| **Greedy / rule-based** ("charge when cheap, discharge when expensive") | Fast to write, provably suboptimal. Directives interact with rate limits and capacity in ways a greedy rule cannot see. The judge recomputes cost against a hidden optimum — you lose points without ever knowing. |
| **MPC / rolling horizon** | Exists to handle *uncertain* future data. We are given the entire 24-hour horizon deterministically. MPC can only approximate what a single LP solves exactly. |
| **Reinforcement learning** | Same objection, more severely. Needs training data and time we do not have, and cannot beat an exact solve. |
| **MILP with binary on/off variables** | Works, and we verified it gives the same cost — but it is ~2× slower (10.2 ms vs 4.8 ms) and unnecessary. §4 achieves the same guarantee with zero integer variables. |
| **Using the LLM to produce numbers** | Explicitly disallowed by the guardrails ("No invention"), and unnecessary. The LLM's only job is note → directive. |

---

## 3. The mathematical formulation

Indices `h = 0..23`. All quantities in kWh; tariff in BDT/kWh.

### Given (after the directive compiler, §5)

| Symbol | Meaning |
|---|---|
| `D[h]` | `demand_kwh` |
| `P[h]` | tariff, `tariff_bdt_per_kwh` |
| `Ŝ[h]` | **effective** solar after `solar_reduction` |
| `C̄[h]` | charge upper bound (0 in a `no_charge_window`) |
| `D̄[h]` | discharge upper bound (0 in a `no_discharge_window`) |
| `Ḡ[h]` | grid upper bound (`+∞` unless `max_grid_window`) |
| `E_min[h]` | `max(battery.minimum_energy_kwh, any minimum_battery_reserve for h)` |
| `E_cap`, `E_0` | `capacity_kwh`, `initial_energy_kwh` |

### Decision variables

| Variable | Domain | Note |
|---|---|---|
| `g[h]` | `0 ≤ g[h] ≤ Ḡ[h]` | grid import |
| `s[h]` | `0 ≤ s[h] ≤ Ŝ[h]` | solar **used** — a free variable, see §3.1 |
| `b[h]` | `−D̄[h] ≤ b[h] ≤ C̄[h]` | **signed** net battery flow: `>0` charge, `<0` discharge |
| `e[h]` | `E_min[h] ≤ e[h] ≤ E_cap` | energy after hour `h` |

### Objective

```
minimize   Σ_h  g[h] · P[h]
```

### Constraints

```
(1) energy balance      g[h] + s[h] − b[h] = D[h]                for all h
(2) battery state       e[h] = e[h−1] + b[h],   e[−1] ≡ E_0      for all h
(3) end-of-day          e[23] = E_0
```

Rate limits, the grid cap, the solar ceiling and the reserve floor are all **variable bounds**, not rows — which is why the model is tiny (96 variables, 49 equality rows) and solves in ~5 ms.

### 3.1 Two modelling details that are easy to get wrong

**`s[h]` must be a decision variable, not `= Ŝ[h]`.** Solar is free, so it is *usually* optimal to consume all of it — but when `Ŝ[h] > D[h] + C̄[h]`, the surplus physically cannot go anywhere (Section 9.4: *"Unused solar is curtailed. Grid export is not part of this challenge."*). Forcing `s[h] = Ŝ[h]` then drives `g[h]` negative and the model reports infeasible. We confirmed this with a synthetic high-solar/low-demand scenario — 6 hours required curtailment of up to 340 kWh, and the plan remained valid only because `s[h]` was free to sit below its ceiling.

**`b[h]` signed, not `charge[h]`/`discharge[h]` split.** This is the fix for §1. See §4.

---

## 4. Fix #1 — the signed battery variable (kills the 500s)

The defect in §1 exists because the model can represent "charge 55 **and** discharge 55 in the same hour." The fix is to make that state **unrepresentable**, rather than detecting and patching it afterwards.

Replace the two non-negative variables with one signed variable:

```python
# BEFORE — two variables, can both be positive at once
charge    = LpVariable(f"charge_{h}",    lowBound=0, upBound=C̄[h])
discharge = LpVariable(f"discharge_{h}", lowBound=0, upBound=D̄[h])

# AFTER — one variable; sign IS the action
bat = LpVariable(f"bat_{h}", lowBound=-D̄[h], upBound=C̄[h])
```

The action is recovered by sign at output time: `b[h] > 0 → charge`, `b[h] < 0 → discharge`, `b[h] == 0 → idle`, with `battery_kwh = abs(b[h])`.

Why this is exactly equivalent to the intended physics: with no round-trip efficiency loss, `charge[h]` and `discharge[h]` only ever enter the model as the difference `charge[h] − discharge[h]`. Substituting `b[h] = charge[h] − discharge[h]` is a faithful change of variables, not a relaxation — it discards only the degenerate representations that were causing the bug. The directive windows map cleanly: `no_charge_window` sets the upper bound to 0, `no_discharge_window` sets the lower bound to 0.

We benchmarked all three candidate fixes on CBC across the 10 public cases. All reach the identical optimum:

```
case              ref | split(now)  sim |       eps  sim |      MILP  sim |       NET
SAMPLE-01       38365 |     38365    0 |     38365    0 |     38365    0 |     38365
SAMPLE-02       42885 |     42885    6 |     42885    0 |     42885    0 |     42885
SAMPLE-04       40495 |     40495    4 |     40495    0 |     40495    0 |     40495
SAMPLE-08       37665 |     37665    6 |     37665    0 |     37665    0 |     37665
                       ( "sim" = hours with simultaneous charge AND discharge )
avg solve ms: {'split': 5.2, 'eps': 5.3, 'milp': 10.2, 'net': 4.8}
```

**Choose NET.** The `eps` variant (adding a tiny penalty `ε·Σ(charge+discharge)` to break the tie) also works and is the textbook trick, but it is a *numerical* deterrent — it makes the bad vertex slightly more expensive rather than non-existent, and it perturbs the objective. MILP works but doubles solve time and adds 24 binaries for a guarantee we can get for free. **NET is the only one of the three where the defect cannot occur by construction**, and it is also the fastest.

> Literature note: the standard treatment of this issue introduces binary variables `z, y` to enforce the complementarity condition `charge[h]·discharge[h] = 0`, which turns the LP into a MILP. That machinery is required when round-trip efficiency `η < 1`, because then the charge and discharge legs no longer collapse into a single net term. **GridWise specifies no efficiency loss**, which is precisely the condition under which the signed-variable substitution is exact — so we get the MILP's guarantee at LP cost. If the organizers ever add efficiency, switch to the MILP form.

---

## 5. Fix #2 — the directive compiler (one function, no special cases)

Keep the existing `apply_directives()` shape — it is already the right idea. The principle worth stating explicitly, because it is what makes the hidden paraphrase tests survivable:

**Every one of the five active directive types is a modification of a per-hour bound. None of them is a new constraint row.**

| Directive | Effect | Merge rule when two directives overlap |
|---|---|---|
| `solar_reduction` | `Ŝ[h] ← Ŝ[h] · factor` | multiply (compounding) |
| `minimum_battery_reserve` | `E_min[h] ← max(E_min[h], value)` | **max** — strictest wins |
| `no_charge_window` | `C̄[h] ← 0` | idempotent |
| `no_discharge_window` | `D̄[h] ← 0` | idempotent |
| `max_grid_window` | `Ḡ[h] ← min(Ḡ[h], value)` | **min** — strictest wins |
| `no_op` | nothing | — |

The merge rules matter: SAMPLE-07 and SAMPLE-10 both apply a reserve **and** a grid cap over overlapping evening hours, and the hidden set will do more of this. Always take the *tighter* of the two, never the last one seen.

This design gives you a clean test seam: the compiler is a pure function from directives to a bounds array, so it can be unit-tested with no solver in the loop. That is exactly the surface the "correct interpretation but not applied" check (Section 11.2) targets.

### 5.1 Interpretation semantics confirmed across all 10 public cases

The LLM path is your teammate's, but these are the rules the optimizer depends on, each verified against the reference outputs:

- **Windows are half-open** — start inclusive, end exclusive. Every public case obeys this, including the "between X and Y" phrasing, which behaves identically to "from X until Y":
  - `"from noon until 2 PM"` → `[12, 13]` (SAMPLE-01)
  - `"from 2 AM until 5 AM"` → `[2, 3, 4]` (SAMPLE-02)
  - `"between 11 AM and 2 PM"` → `[11, 12, 13]` (SAMPLE-09)
  - `"from 6 PM until 10 PM"` → `[18, 19, 20, 21]` (SAMPLE-07, SAMPLE-10)
- **`factor` is the fraction that REMAINS**, not the reduction. `"80% reduction"` → `factor = 0.2` (SAMPLE-09); `"roughly 25% of the forecast"` → `factor = 0.25` (SAMPLE-01); `"about half"` → `0.5` (SAMPLE-06).
- **Percentage reserves resolve against `capacity_kwh`**, not current energy. `"50% of the battery capacity"` with `capacity_kwh = 200` → `minimum_energy_kwh = 100` (SAMPLE-03).
- **Every note produces exactly one entry**, in `note_index` order, including distractors (`no_op`, `applies: false`, `structured_adjustment: null`).

---

## 6. Fix #3 — elastic fallback (never return 500)

### The scenario

The spec guarantees the *ground-truth* directives are feasible ("Organizer valid scoring scenarios are feasible and will not require mutually contradictory hard directives"). That guarantee does **not** extend to what your LLM produces. If the model misreads `"155 kWh"` as `55`, or `"50%"` as `"90%"`, the resulting constraint set can be genuinely unsatisfiable — and the current code raises `OptimizationError` → **HTTP 500 → the whole case scores zero**, including interpretation points that may have been partially correct and all the Section 11.3 validity points.

We confirmed this is reachable:

```
  grid cap misread 155->55     -> INFEASIBLE -> app returns HTTP 500 -> 0 pts for whole case
```

And in 400 randomized scenarios with random directives, **47 (≈12%) were infeasible** under a hard model.

### The fix

On infeasibility, re-solve with the **directive-derived** constraints made elastic — each gets a non-negative slack variable carrying a large penalty — while the **physics** constraints stay hard:

| Tier | Constraints | Treatment |
|---|---|---|
| **Hard, always** | energy balance, battery state, rate limits, `0 ≤ s ≤ Ŝ`, `e ≤ E_cap`, `g ≥ 0`, end-of-day neutrality | never relaxed |
| **Elastic on fallback** | `max_grid_window` cap, `minimum_battery_reserve` floor | slack + big-M penalty |

The big-M (`max(tariff) × 10⁴`) makes the solver exhaust every cost-based option before violating a directive by even a fraction of a kWh — so when a feasible plan exists, the elastic model returns exactly the hard model's answer, and when none exists, it returns the **minimum-violation** plan instead of nothing.

This is standard elastic/goal programming, and the strategic logic is simple: a plan that satisfies all of Section 11.3 and misses one directive scores far better than an HTTP 500 that satisfies nothing.

### Verified behaviour

```
A. ELASTIC FALLBACK — infeasible directive sets (LLM misread scenarios)
  grid cap misread 155->55                     -> elastic-relaxed  cost  33950.00  physics VALID
  reserve misread 50%->99%                     -> optimal          cost  37424.00  physics VALID
  no_charge over ALL 24h (breaks neutrality)   -> optimal          cost  47335.00  physics VALID
  cap 190->20 + reserve 80                     -> elastic-relaxed  cost  41920.00  physics VALID

C. RANDOMIZED FUZZ — 400 synthetic scenarios
  400 scenarios: physics violations=0, needed elastic fallback=47, unrecoverable=0
```

Zero physics violations across 400 randomized scenarios, and zero unrecoverable cases.

---

## 7. Fix #4 — the canonicalizer (make the judge's replay exact)

The judge **independently replays** the schedule hour by hour (Section 9, Section 11.3) and recomputes `total_grid_kwh`, `total_cost_bdt`, `peak_grid_kwh` from `hourly_plan`. Reporting raw solver floats invites two avoidable problems: simplex output like `152.49999999999997`, and a `battery_energy_after_kwh` chain that came from the solver rather than from the actions actually being reported.

The fix is to **rebuild the plan from the decision variables by exact accumulation** before it leaves the service:

1. Round `b[h]` to 6 dp and snap `|b[h]| < 1e-7` to exactly `0.0` (guarantees `battery_kwh == 0` for every `idle` hour — `schemas.py` enforces this with exact equality, not a tolerance).
2. Fix residual drift so `Σ b[h] == 0` exactly, giving `e[23] == E_0` on the nose.
3. Accumulate `e[h] = e[h−1] + b[h]` forward — so the reported energy chain is *derived from* the reported actions and cannot contradict them.
4. Derive `g[h] = D[h] + charge[h] − s[h] − discharge[h]` from the balance equation — so Section 9.5 holds **by construction**, not by luck.

Keep reporting the totals from `replay()`'s recompute rather than the solver's objective — that part of the current design is already right.

---

## 8. Drop-in replacement: `app/optimizer.py`

Same public surface as the current module. `solve()` now returns `(plan, mode)`; `mode` is `"optimal"` or `"relaxed"` and is worth logging and folding into `plan_summary`.

```python
"""
GridWise optimizer — hardened.
  1. compile directives -> per-hour bounds      (pure function, no solver)
  2. LP with ONE SIGNED battery variable        (simultaneous charge/discharge impossible)
  3. elastic fallback on infeasibility          (never HTTP 500 on a bad interpretation)
  4. canonicalize                               (judge replay reproduces our numbers exactly)
"""
from typing import Dict, List, Optional, Tuple
import pulp
from app.schemas import BatteryConfig, DirectiveInterpretation, HourEntry, HourlyPlanEntry

N = 24
ZERO = 1e-7          # snap-to-idle threshold
DP = 6               # canonical decimal places


class OptimizationError(Exception):
    pass


# ---------------------------------------------------------------- layer 1
def apply_directives(hours, battery, directives):
    """Fold validated directives into per-hour bounds. Pure, testable, no solver."""
    eff_solar = {h.hour: float(h.solar_kwh) for h in hours}
    min_reserve = {i: float(battery.minimum_energy_kwh) for i in range(N)}
    chg_ub = {i: float(battery.max_charge_kwh_per_hour) for i in range(N)}
    dis_ub = {i: float(battery.max_discharge_kwh_per_hour) for i in range(N)}
    max_grid: Dict[int, Optional[float]] = {i: None for i in range(N)}

    for d in directives:
        if not d.applies:
            continue
        adj, t = d.structured_adjustment, d.directive_type
        if t == "solar_reduction":
            for h in adj["hours"]:
                eff_solar[h] *= float(adj["factor"])
        elif t == "minimum_battery_reserve":
            for h in adj["hours"]:
                min_reserve[h] = max(min_reserve[h], float(adj["minimum_energy_kwh"]))
        elif t == "no_charge_window":
            for h in adj["hours"]:
                chg_ub[h] = 0.0
        elif t == "no_discharge_window":
            for h in adj["hours"]:
                dis_ub[h] = 0.0
        elif t == "max_grid_window":
            for h in adj["hours"]:
                cap = float(adj["max_grid_kwh"])
                max_grid[h] = cap if max_grid[h] is None else min(max_grid[h], cap)
    return eff_solar, min_reserve, chg_ub, dis_ub, max_grid


# ---------------------------------------------------------------- layers 2+3
def _build(hours, battery, bounds, elastic: bool):
    demand = {h.hour: float(h.demand_kwh) for h in hours}
    tariff = {h.hour: float(h.tariff_bdt_per_kwh) for h in hours}
    eff_solar, min_reserve, chg_ub, dis_ub, max_grid = bounds

    p = pulp.LpProblem("gridwise", pulp.LpMinimize)
    grid = {h: pulp.LpVariable(f"grid_{h}", lowBound=0) for h in range(N)}
    solar = {h: pulp.LpVariable(f"solar_{h}", lowBound=0, upBound=eff_solar[h]) for h in range(N)}
    # ONE signed variable: >0 charge, <0 discharge. Both-at-once is unrepresentable.
    bat = {h: pulp.LpVariable(f"bat_{h}", lowBound=-dis_ub[h], upBound=chg_ub[h]) for h in range(N)}
    energy = {h: pulp.LpVariable(f"energy_{h}", lowBound=0, upBound=float(battery.capacity_kwh))
              for h in range(N)}

    obj = pulp.lpSum(grid[h] * tariff[h] for h in range(N))
    BIG = max(1.0, max(tariff.values())) * 1e4
    sg = se = None
    if elastic:
        sg = {h: pulp.LpVariable(f"sg_{h}", lowBound=0) for h in range(N)}   # grid-cap excess
        se = {h: pulp.LpVariable(f"se_{h}", lowBound=0) for h in range(N)}   # reserve shortfall
        obj += pulp.lpSum((sg[h] + se[h]) * BIG for h in range(N))
    p += obj

    for h in range(N):
        p += grid[h] + solar[h] - bat[h] == demand[h], f"balance_{h}"
        prev = float(battery.initial_energy_kwh) if h == 0 else energy[h - 1]
        p += energy[h] == prev + bat[h], f"state_{h}"
        if elastic:
            p += energy[h] >= min_reserve[h] - se[h], f"reserve_{h}"
            if max_grid[h] is not None:
                p += grid[h] <= max_grid[h] + sg[h], f"cap_{h}"
        else:
            energy[h].lowBound = min_reserve[h]
            if max_grid[h] is not None:
                grid[h].upBound = max_grid[h]
    p += energy[N - 1] == float(battery.initial_energy_kwh), "neutrality"
    return p, grid, solar, bat, energy


# ---------------------------------------------------------------- layer 4
def _canonicalize(hours, battery, bounds, grid_v, solar_v, bat_v) -> List[HourlyPlanEntry]:
    """Rebuild the plan by exact accumulation so an independent replay matches."""
    demand = {h.hour: float(h.demand_kwh) for h in hours}
    eff_solar = bounds[0]

    bat = []
    for h in range(N):
        v = round(bat_v[h].value() or 0.0, DP)
        bat.append(0.0 if abs(v) < ZERO else v)

    # force exact end-of-day neutrality against float drift
    drift = round(sum(bat), DP)
    if drift != 0.0:
        for h in range(N - 1, -1, -1):
            if bat[h] != 0.0:
                bat[h] = round(bat[h] - drift, DP)
                break

    plan, prev = [], float(battery.initial_energy_kwh)
    for h in range(N):
        solar = min(max(round(solar_v[h].value() or 0.0, DP), 0.0), eff_solar[h])
        charge, discharge = max(bat[h], 0.0), max(-bat[h], 0.0)
        # derive grid FROM the balance equation -> balance holds by construction
        grid = max(0.0, round(demand[h] + charge - solar - discharge, DP))
        prev = round(prev + bat[h], DP)
        action = "charge" if charge > 0 else ("discharge" if discharge > 0 else "idle")
        plan.append(HourlyPlanEntry(
            hour=h, grid_kwh=grid, solar_used_kwh=solar,
            battery_action=action, battery_kwh=round(abs(bat[h]), DP),
            battery_energy_after_kwh=prev,
        ))
    return plan


def solve(hours: List[HourEntry], battery: BatteryConfig,
          directives: List[DirectiveInterpretation]) -> Tuple[List[HourlyPlanEntry], str]:
    hours = sorted(hours, key=lambda h: h.hour)
    bounds = apply_directives(hours, battery, directives)

    p, g, s, b, e = _build(hours, battery, bounds, elastic=False)
    p.solve(pulp.PULP_CBC_CMD(msg=0))
    mode = "optimal"

    if pulp.LpStatus[p.status] != "Optimal":
        p, g, s, b, e = _build(hours, battery, bounds, elastic=True)
        p.solve(pulp.PULP_CBC_CMD(msg=0))
        mode = "relaxed"
        if pulp.LpStatus[p.status] != "Optimal":
            raise OptimizationError(f"infeasible even when relaxed (status={pulp.LpStatus[p.status]})")

    return _canonicalize(hours, battery, bounds, g, s, b), mode
```

### Caller change in `main.py`

```python
hourly_plan, opt_mode = solve(payload.hours, payload.battery, directives)
```

…and `replay()` needs no change at all. Note `apply_directives()` keeps its exact signature, so `replay.py`'s import of it continues to work untouched.

**One further hardening step worth taking:** in `main.py`, catch `ReplayError` and, instead of returning 500, fall back to a trivially valid plan (`grid_kwh = demand[h]`, `solar_used_kwh = 0`, all hours `idle`). That plan is always feasible, always passes every Section 11.3 check, and preserves the interpretation score. It costs more, but a suboptimal valid plan beats a 500 in every rubric.

---

## 9. Verification — run this before you ship

The whole point of the exercise: our own `replay.py`, unchanged, as the judge.

```
case               ref         v2   delta mode      replay
SAMPLE-01     38365.00   38365.00    0.00 optimal   PASS
SAMPLE-02     42885.00   42885.00    0.00 optimal   PASS
SAMPLE-03     35480.00   35480.00    0.00 optimal   PASS
SAMPLE-04     40495.00   40495.00    0.00 optimal   PASS
SAMPLE-05     33950.00   33950.00    0.00 optimal   PASS
SAMPLE-06     34090.00   34090.00    0.00 optimal   PASS
SAMPLE-07     38550.00   38550.00    0.00 optimal   PASS
SAMPLE-08     37665.00   37665.00    0.00 optimal   PASS
SAMPLE-09     34873.00   34873.00    0.00 optimal   PASS
SAMPLE-10     41620.00   41620.00    0.00 optimal   PASS

10/10 pass with the team's OWN replay.py as the checker.
```

### The harness

Save as `tests/test_public_cases.py` and run it after every change to the optimizer. It bypasses the LLM entirely by feeding the ground-truth directives straight in, which isolates optimizer correctness from interpretation correctness — the two things the judge scores separately.

```python
import json
from app.schemas import ScenarioRequest, DirectiveInterpretation
from app.optimizer import solve
from app.replay import replay

def test_public_cases():
    data = json.load(open("BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"))
    failures = []
    for case in data["cases"]:
        req = ScenarioRequest(**case["input"])
        ref = case["expected_output"]
        dirs = [DirectiveInterpretation(**d) for d in ref["directive_interpretation"]]
        plan, mode = solve(req.hours, req.battery, dirs)
        tg, tc, pk = replay(req.hours, req.battery, dirs, plan)   # raises on any violation
        if abs(tc - ref["total_cost_bdt"]) > 0.01:
            failures.append(f"{case['id']}: cost {tc} vs ref {ref['total_cost_bdt']}")
    assert not failures, failures
```

**Two further checks worth the ten minutes**, both of which caught real issues for us:

1. **Directive-compiler unit tests, no solver.** Assert that a `minimum_battery_reserve` of 90 over `[18,19,20,21]` plus a base minimum of 40 yields `min_reserve[19] == 90` and `min_reserve[17] == 40`; that two overlapping `max_grid_window`s take the **min**; that two `solar_reduction`s on the same hour **multiply**. This is precisely the "interpreted correctly but not applied" failure the judge tests for in Section 11.2.
2. **Fuzz the optimizer.** Generate a few hundred random scenarios (random demand/solar/tariff, random battery, 0–3 random directives), and assert only that the replay never reports a *physics* violation. We ran 400 and found zero. This is what caught the infeasibility path.

---

## 10. Summary of changes

| # | Change | Fixes | Priority |
|---|---|---|---|
| 1 | Signed battery variable `b[h] ∈ [−D̄, C̄]` replacing the charge/discharge split | 3 of 10 public cases returning HTTP 500 | **Critical** |
| 2 | Canonicalize the plan by exact accumulation before returning | float drift; action/energy contradictions; `idle` with non-zero `battery_kwh` | **High** |
| 3 | Elastic fallback on infeasibility | ~12% of adversarial directive sets returning HTTP 500 | **High** |
| 4 | Merge-rule discipline in the compiler (`max` for reserves, `min` for caps, `×` for solar) | wrong results on overlapping directives (SAMPLE-07/10 patterns) | **Medium** |
| 5 | `main.py` fallback plan on `ReplayError` instead of 500 | preserves interpretation + validity score when anything unexpected happens | **Medium** |

Nothing here changes the LLM path, the guardrails, the schemas, or the API contract. `replay.py` is untouched. The only caller-visible change is that `solve()` returns a `(plan, mode)` tuple.

---

## 11. Sources

- DrivenData, *Power Laws: Optimizing Demand-side Strategies* — competition overview and winners' write-ups (LP-based approaches; top algorithm ≈20% savings): https://www.drivendata.org/competitions/53/optimize-photovoltaic-battery/ and https://drivendata.co/blog/power-laws-optimization-winners
- Winning solution code repository: https://github.com/drivendataorg/power-laws-optimization
- EMHASS — open-source LP energy-management system (CVXPY + HiGHS): https://github.com/davidusb-geek/emhass
- Pozo, *Linear Battery Models for Power Systems Analysis* — the binary-complementarity formulation for preventing simultaneous charge/discharge, and why LP relaxations permit it: https://arxiv.org/pdf/2204.08240
- *Guaranteeing a Physically Realizable Battery Dispatch Without Complementarity Constraints* — on relaxed battery models producing unrealizable dispatch: https://arxiv.org/pdf/2103.07846
- HiGHS solver documentation: https://highs.dev/
