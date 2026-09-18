"""
Deterministic LP optimizer. Operator directives are applied here as
constraints/parameters — never trusted math from the LLM, only validated
structured_adjustment objects that guardrails.py has already accepted.

  1. apply_directives / compile_bounds  directives -> per-hour bounds (pure, no solver)
  2. LP with ONE signed battery variable  (simultaneous charge+discharge is unrepresentable)
  3. elastic fallback on infeasibility    (directive constraints relaxed, physics stays hard)
  4. canonicalize                         (plan rebuilt by exact accumulation)
"""
import math
from typing import Dict, List, NamedTuple, Optional, Tuple

import pulp

from app.schemas import BatteryConfig, DirectiveInterpretation, HourEntry, HourlyPlanEntry

N = 24
ZERO = 1e-7  # |battery flow| below this is snapped to exactly idle
DP = 6  # canonical decimal places


class OptimizationError(Exception):
    pass


def _nz(x: float) -> float:
    """Normalise negative zero to +0.0 (max(-0.0, 0.0) is -0.0, which JSON would print as "-0.0")."""
    return x + 0.0


# ---------------------------------------------------------------- layer 1
def apply_directives(
    hours: List[HourEntry],
    battery: BatteryConfig,
    directives: List[DirectiveInterpretation],
):
    """Turn validated directives into per-hour optimizer parameters.

    Returns (effective_solar, min_reserve, no_charge_hours, no_discharge_hours, max_grid).
    The shape is a contract with app/replay.py: the two window slots are *sets of hours*.

    Merge rules when directives overlap: solar_reduction multiplies, minimum_battery_reserve
    takes the max, max_grid_window takes the min, no-charge/no-discharge windows union.
    """
    effective_solar = {h.hour: float(h.solar_kwh) for h in hours}
    min_reserve = {i: float(battery.minimum_energy_kwh) for i in range(N)}
    no_charge_hours = set()
    no_discharge_hours = set()
    max_grid: Dict[int, Optional[float]] = {i: None for i in range(N)}

    for d in directives:
        if not d.applies:
            continue
        adj = d.structured_adjustment
        if d.directive_type == "solar_reduction":
            for h in adj["hours"]:
                effective_solar[h] *= float(adj["factor"])
        elif d.directive_type == "minimum_battery_reserve":
            for h in adj["hours"]:
                min_reserve[h] = max(min_reserve[h], float(adj["minimum_energy_kwh"]))
        elif d.directive_type == "no_charge_window":
            no_charge_hours.update(adj["hours"])
        elif d.directive_type == "no_discharge_window":
            no_discharge_hours.update(adj["hours"])
        elif d.directive_type == "max_grid_window":
            cap = float(adj["max_grid_kwh"])
            for h in adj["hours"]:
                current = max_grid[h]
                max_grid[h] = cap if current is None else min(current, cap)

    return effective_solar, min_reserve, no_charge_hours, no_discharge_hours, max_grid


class Bounds(NamedTuple):
    eff_solar: Dict[int, float]
    min_reserve: Dict[int, float]
    chg_ub: Dict[int, float]  # 0 inside a no_charge_window
    dis_ub: Dict[int, float]  # 0 inside a no_discharge_window
    max_grid: Dict[int, Optional[float]]


def compile_bounds(
    hours: List[HourEntry],
    battery: BatteryConfig,
    directives: List[DirectiveInterpretation],
) -> Bounds:
    """apply_directives() expressed as per-hour numeric bounds. Pure, solver-free."""
    eff_solar, min_reserve, no_charge, no_discharge, max_grid = apply_directives(hours, battery, directives)
    chg_ub = {i: 0.0 if i in no_charge else float(battery.max_charge_kwh_per_hour) for i in range(N)}
    dis_ub = {i: 0.0 if i in no_discharge else float(battery.max_discharge_kwh_per_hour) for i in range(N)}
    return Bounds(eff_solar, min_reserve, chg_ub, dis_ub, max_grid)


# ---------------------------------------------------------------- layers 2+3
def _build(hours, battery, bounds: Bounds, elastic: bool):
    demand = {h.hour: float(h.demand_kwh) for h in hours}
    tariff = {h.hour: float(h.tariff_bdt_per_kwh) for h in hours}
    cap = float(battery.capacity_kwh)
    e0 = float(battery.initial_energy_kwh)
    base_min = float(battery.minimum_energy_kwh)

    prob = pulp.LpProblem("gridwise_energy", pulp.LpMinimize)
    grid = {h: pulp.LpVariable(f"grid_{h}", lowBound=0) for h in range(N)}
    # Free up to the effective ceiling: unusable surplus solar is curtailed, not forced onto the grid balance.
    solar = {h: pulp.LpVariable(f"solar_{h}", lowBound=0, upBound=bounds.eff_solar[h]) for h in range(N)}
    # ONE signed variable: >0 charge, <0 discharge. Both-at-once cannot be expressed.
    bat = {h: pulp.LpVariable(f"bat_{h}", lowBound=-bounds.dis_ub[h], upBound=bounds.chg_ub[h]) for h in range(N)}
    energy = {h: pulp.LpVariable(f"energy_{h}", lowBound=base_min, upBound=cap) for h in range(N)}

    obj = pulp.lpSum(grid[h] * tariff[h] for h in range(N))
    if elastic:
        big = max(1.0, max(tariff.values())) * 1e4
        s_grid = {h: pulp.LpVariable(f"slack_grid_{h}", lowBound=0) for h in range(N)}
        s_res = {h: pulp.LpVariable(f"slack_reserve_{h}", lowBound=0) for h in range(N)}
        obj += pulp.lpSum((s_grid[h] + s_res[h]) * big for h in range(N))
    prob += obj

    for h in range(N):
        prob += grid[h] + solar[h] - bat[h] == demand[h], f"balance_{h}"
        prev = e0 if h == 0 else energy[h - 1]
        prob += energy[h] == prev + bat[h], f"state_{h}"
        if elastic:
            # Directive-derived limits become soft; the base battery minimum stays a hard variable bound.
            if bounds.min_reserve[h] > base_min:
                prob += energy[h] >= bounds.min_reserve[h] - s_res[h], f"reserve_{h}"
            if bounds.max_grid[h] is not None:
                prob += grid[h] <= bounds.max_grid[h] + s_grid[h], f"cap_{h}"
        else:
            energy[h].lowBound = bounds.min_reserve[h]
            if bounds.max_grid[h] is not None:
                grid[h].upBound = bounds.max_grid[h]
    prob += energy[N - 1] == e0, "end_of_day_neutrality"
    return prob, grid, solar, bat, energy


# ---------------------------------------------------------------- layer 4
def _canonicalize(hours, battery, bounds: Bounds, solar_v, bat_v) -> List[HourlyPlanEntry]:
    """Rebuild the plan from raw solver values by exact accumulation.

    bat_v / solar_v are plain float lists indexed by hour. Grid is derived from the balance
    equation and the energy chain from the reported actions, so both are consistent by construction.
    """
    eff_solar, chg_ub, dis_ub = bounds.eff_solar, bounds.chg_ub, bounds.dis_ub
    demand = {h.hour: float(h.demand_kwh) for h in hours}
    e0 = float(battery.initial_energy_kwh)

    bat = []
    for v in bat_v:
        v = round(v, DP)
        bat.append(0.0 if abs(v) < ZERO else v)

    # Push residual rounding drift into one flow so the day nets to exactly zero. Prefer an hour where the
    # correction keeps the rate limit, direction (no-charge / no-discharge windows), balance (discharge <=
    # demand + charge) AND every later energy value inside [minimum, capacity]; degrade gracefully if none.
    drift = round(math.fsum(bat), DP)
    if drift != 0.0:
        base_min, cap = float(battery.minimum_energy_kwh), float(battery.capacity_kwh)
        chain, run = [], e0
        for b in bat:
            run += b
            chain.append(run)

        def flow_ok(i):
            return (
                bat[i] != 0.0
                and (bat[i] > 0) == (bat[i] - drift > 0)
                and -dis_ub[i] <= bat[i] - drift <= chg_ub[i]
                and demand[i] + bat[i] - drift >= 0.0
            )

        def energy_ok(i):
            return all(base_min <= chain[j] - drift <= cap for j in range(i, N))

        tiers = (
            [i for i in range(N) if flow_ok(i) and energy_ok(i)],
            [i for i in range(N) if flow_ok(i)],
            list(range(N)),
        )
        k = max(next(t for t in tiers if t), key=lambda i: abs(bat[i]))
        bat[k] = round(bat[k] - drift, DP)

    plan, prev = [], e0
    for h in range(N):
        charge, discharge = max(bat[h], 0.0), max(-bat[h], 0.0)
        s = min(max(round(solar_v[h], DP), 0.0), eff_solar[h])
        g = round(demand[h] + charge - s - discharge, DP)
        if g < 0.0:  # rounding noise only: give the curtailable solar back rather than break the balance
            s = max(0.0, round(s + g, DP))
            g = max(0.0, round(demand[h] + charge - s - discharge, DP))

        prev = round(prev + bat[h], 9)
        if h == N - 1 and abs(prev - e0) < 1e-6:
            prev = e0
        action = "charge" if charge > 0 else ("discharge" if discharge > 0 else "idle")
        plan.append(
            HourlyPlanEntry(
                hour=h,
                grid_kwh=_nz(g),
                solar_used_kwh=_nz(s),
                battery_action=action,
                battery_kwh=_nz(abs(bat[h])),
                battery_energy_after_kwh=_nz(max(0.0, prev)),
            )
        )
    return plan


def _vals(vs: Dict[int, pulp.LpVariable]) -> List[float]:
    return [vs[h].value() or 0.0 for h in range(N)]


STAGE2_SLACK = 1e-7  # BDT of slack on the stage-2 cost constraint: solver-tolerance level only, the solver spends it


def solve(
    hours: List[HourEntry],
    battery: BatteryConfig,
    directives: List[DirectiveInterpretation],
    minimize_peak: bool = True,
) -> Tuple[List[HourlyPlanEntry], str]:
    """Returns (hourly_plan, mode). mode is "optimal", or "relaxed" when the directive set was
    infeasible and the minimum-violation plan is returned instead.

    Two-stage objective when optimal: (1) minimise grid cost; (2) among plans of that cost, minimise the peak
    hourly grid draw. minimize_peak=False returns the stage-1 plan alone."""
    hours = sorted(hours, key=lambda h: h.hour)
    bounds = compile_bounds(hours, battery, directives)

    prob, grid, solar, bat, energy = _build(hours, battery, bounds, elastic=False)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    if pulp.LpStatus[prob.status] == "Optimal":
        plan = _canonicalize(hours, battery, bounds, _vals(solar), _vals(bat))
        if minimize_peak:
            try:  # stage 2 is an optional refinement: nothing it does may break the stage-1 plan
                plan = _min_peak_plan(hours, battery, bounds, plan, pulp.value(prob.objective) or 0.0) or plan
            except Exception:  # noqa: BLE001
                pass
        return plan, "optimal"

    prob, grid, solar, bat, energy = _build(hours, battery, bounds, elastic=True)
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        raise OptimizationError(f"infeasible even when relaxed (status={pulp.LpStatus[prob.status]})")
    return _canonicalize(hours, battery, bounds, _vals(solar), _vals(bat)), "relaxed"


def plan_cost(hours: List[HourEntry], plan: List[HourlyPlanEntry]) -> float:
    tariff = {h.hour: float(h.tariff_bdt_per_kwh) for h in hours}
    return math.fsum(p.grid_kwh * tariff[p.hour] for p in plan)


COST_TOL = 1e-3  # stage 2 is accepted only if it costs no more than stage 1 by this much (judge tolerance: 0.01)


def _min_peak_plan(hours, battery, bounds: Bounds, stage1: List[HourlyPlanEntry],
                   c_star: float) -> Optional[List[HourlyPlanEntry]]:
    """Stage 2: among plans that cost no more than stage 1, minimise peak grid draw.

    Cost stays primary: the stage-2 model is constrained to stage 1's cost, and its canonical plan is used only
    if its recomputed cost is within COST_TOL of stage 1's. Anything else (infeasible, numerically noisy, worse
    cost, or a physics-invalid rebuild) returns None and the stage-1 plan stands.
    """
    c1 = plan_cost(hours, stage1)
    tariff = {h.hour: float(h.tariff_bdt_per_kwh) for h in hours}  # c_star: stage 1's raw LP optimum
    prob, grid, solar, bat, energy = _build(hours, battery, bounds, elastic=False)
    peak = pulp.LpVariable("peak", lowBound=0)
    prob.objective = pulp.LpAffineExpression([(peak, 1)])
    prob += pulp.lpSum(grid[h] * tariff[h] for h in range(N)) <= c_star + STAGE2_SLACK, "stage1_cost"
    for h in range(N):
        prob += grid[h] <= peak, f"peak_{h}"
    prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[prob.status] != "Optimal":
        return None
    plan2 = _canonicalize(hours, battery, bounds, _vals(solar), _vals(bat))
    if plan_cost(hours, plan2) > c1 + COST_TOL:
        return None
    if max(p.grid_kwh for p in plan2) > max(p.grid_kwh for p in stage1):  # never trade a higher peak for noise
        return None
    return plan2


def trivial_plan(hours: List[HourEntry], battery: BatteryConfig) -> List[HourlyPlanEntry]:
    """Grid serves all demand, solar and battery untouched. Satisfies every physics rule by construction;
    it does NOT necessarily satisfy directive limits (e.g. a grid cap below demand)."""
    e0 = float(battery.initial_energy_kwh)
    return [
        HourlyPlanEntry(
            hour=h.hour,
            grid_kwh=_nz(float(h.demand_kwh)),
            solar_used_kwh=0.0,
            battery_action="idle",
            battery_kwh=0.0,
            battery_energy_after_kwh=e0,
        )
        for h in sorted(hours, key=lambda x: x.hour)
    ]
