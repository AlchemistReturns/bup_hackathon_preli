"""
Hybrid MILP + DP optimizer for GridWise 24-hour energy scheduling.

Phase 1 — DP Scout  (~5ms): Fast feasible solution + battery reachability bounds.
Phase 2 — MILP Refine (~10-30ms): Exact optimal with DP warm start + tight bounds.
Phase 3 — Cross-Validate: Compare DP vs MILP costs, pick best, flag bugs.

Falls back to DP if MILP solver encounters issues.
"""
import logging
import math
from typing import Dict, List, Optional, Set, Tuple

import pulp

from app.schemas import BatteryConfig, DirectiveInterpretation, HourEntry, HourlyPlanEntry

logger = logging.getLogger("gridwise.optimizer")

EPS = 1e-6
CROSS_VALIDATE_TOL = 2.0  # BDT tolerance for DP vs MILP cost agreement


class OptimizationError(Exception):
    pass


# ──────────────────────────────────────────────────────────────────────
# Shared: directive application (also used by replay.py)
# ──────────────────────────────────────────────────────────────────────

def apply_directives(
    hours: List[HourEntry],
    battery: BatteryConfig,
    directives: List[DirectiveInterpretation],
):
    """Turn validated directives into per-hour optimizer parameters."""
    n = 24
    effective_solar = {h.hour: h.solar_kwh for h in hours}
    min_reserve = {i: battery.minimum_energy_kwh for i in range(n)}
    no_charge_hours = set()
    no_discharge_hours = set()
    max_grid = {i: None for i in range(n)}

    for d in directives:
        if not d.applies:
            continue
        adj = d.structured_adjustment
        if d.directive_type == "solar_reduction":
            for h in adj["hours"]:
                effective_solar[h] = effective_solar[h] * adj["factor"]
        elif d.directive_type == "minimum_battery_reserve":
            for h in adj["hours"]:
                min_reserve[h] = max(min_reserve[h], adj["minimum_energy_kwh"])
        elif d.directive_type == "no_charge_window":
            no_charge_hours.update(adj["hours"])
        elif d.directive_type == "no_discharge_window":
            no_discharge_hours.update(adj["hours"])
        elif d.directive_type == "max_grid_window":
            for h in adj["hours"]:
                current = max_grid[h]
                max_grid[h] = adj["max_grid_kwh"] if current is None else min(current, adj["max_grid_kwh"])

    return effective_solar, min_reserve, no_charge_hours, no_discharge_hours, max_grid


# ──────────────────────────────────────────────────────────────────────
# Phase 1: DP solver with reachability export
# ──────────────────────────────────────────────────────────────────────

def _solve_dp(
    hours: List[HourEntry],
    battery: BatteryConfig,
    effective_solar: Dict[int, float],
    min_reserve: Dict[int, float],
    no_charge_hours: Set[int],
    no_discharge_hours: Set[int],
    max_grid: Dict[int, Optional[float]],
    granularity: float = 0.5,
) -> Tuple[Optional[List[HourlyPlanEntry]], float, List[Dict[str, float]]]:
    """
    DP solver with configurable granularity.

    Returns:
        (hourly_plan, total_cost, reachability_bounds)
        hourly_plan is None if no feasible schedule exists.
        reachability_bounds[h] = {"min": ..., "max": ...} for battery energy at hour h.
    """
    H = 24
    cap = battery.capacity_kwh
    E0 = battery.initial_energy_kwh
    Emin_base = battery.minimum_energy_kwh
    maxC = battery.max_charge_kwh_per_hour
    maxD = battery.max_discharge_kwh_per_hour

    demand = {h.hour: h.demand_kwh for h in hours}
    tariff = {h.hour: h.tariff_bdt_per_kwh for h in hours}

    # Discretize: convert continuous energy range to integer indices
    step = granularity
    # number of discrete levels
    num_levels = int(round((cap - Emin_base) / step)) + 1

    def e_to_idx(e: float) -> int:
        return int(round((e - Emin_base) / step))

    def idx_to_e(idx: int) -> float:
        return Emin_base + idx * step

    INF = float("inf")

    # dp[h][idx] = minimum cost to reach energy level idx at end of hour h
    dp = [[INF] * num_levels for _ in range(H)]
    # parent[h][idx] = (prev_idx,) for backtracking
    parent = [[(-1,) for _ in range(num_levels)] for _ in range(H)]

    # reachability bounds for MILP tightening
    reachability = [{"min": cap, "max": Emin_base} for _ in range(H)]

    for h in range(H):
        d_h = demand[h]
        t_h = tariff[h]
        sol_h = effective_solar[h]
        min_e_h = min_reserve[h]
        gc = max_grid[h]
        can_charge = h not in no_charge_hours
        can_discharge = h not in no_discharge_hours

        # min and max idx for E_after at this hour
        min_idx_after = max(0, e_to_idx(min_e_h)) if min_e_h > Emin_base else 0
        # make sure min_idx_after is valid
        min_idx_after = max(0, min(min_idx_after, num_levels - 1))

        for idx_before in range(num_levels):
            e_before = idx_to_e(idx_before)
            if e_before < Emin_base - EPS or e_before > cap + EPS:
                continue

            if h == 0:
                if abs(e_before - E0) > EPS:
                    continue
                prev_cost = 0.0
            else:
                prev_cost = dp[h - 1][idx_before]
                if prev_cost >= INF:
                    continue

            # Range of reachable E_after indices
            e_max_after = min(cap, e_before + maxC)
            e_min_after = max(min_e_h, e_before - maxD)
            # Also bound by absolute limits
            e_min_after = max(e_min_after, Emin_base)

            idx_min_after = max(min_idx_after, e_to_idx(max(Emin_base, e_min_after)))
            idx_max_after = min(num_levels - 1, e_to_idx(min(cap, e_max_after)))

            # Clamp
            idx_min_after = max(0, idx_min_after)
            idx_max_after = min(num_levels - 1, idx_max_after)

            for idx_after in range(idx_min_after, idx_max_after + 1):
                e_after = idx_to_e(idx_after)
                delta = e_after - e_before

                charge_amt = max(0.0, delta)
                discharge_amt = max(0.0, -delta)

                # Enforce no-charge / no-discharge
                if charge_amt > EPS and not can_charge:
                    continue
                if discharge_amt > EPS and not can_discharge:
                    continue

                # Enforce rate limits
                if charge_amt > maxC + EPS:
                    continue
                if discharge_amt > maxD + EPS:
                    continue

                # Compute optimal solar usage: maximize solar to minimize grid
                # Energy balance: grid + solar_used + discharge = demand + charge
                # => grid = demand + charge - solar_used - discharge
                # minimize grid => maximize solar_used
                needed_from_solar_and_grid = d_h + charge_amt - discharge_amt
                solar_used = min(sol_h, max(0.0, needed_from_solar_and_grid))
                grid = needed_from_solar_and_grid - solar_used

                if grid < -EPS:
                    continue

                grid = max(0.0, grid)

                # Enforce grid cap
                if gc is not None and grid > gc + EPS:
                    continue

                cost = prev_cost + grid * t_h
                if cost < dp[h][idx_after]:
                    dp[h][idx_after] = cost
                    parent[h][idx_after] = (idx_before,)

                # Update reachability
                reachability[h]["min"] = min(reachability[h]["min"], e_after)
                reachability[h]["max"] = max(reachability[h]["max"], e_after)

    # Find answer: dp[23][idx(E0)]
    final_idx = e_to_idx(E0)
    if final_idx < 0 or final_idx >= num_levels or dp[H - 1][final_idx] >= INF:
        return None, INF, reachability

    total_cost = dp[H - 1][final_idx]

    # Backtrack
    plan: List[HourlyPlanEntry] = []
    idx = final_idx
    path = []
    for h in range(H - 1, -1, -1):
        prev_idx = parent[h][idx][0]
        path.append((h, prev_idx, idx))
        idx = prev_idx

    path.reverse()

    for h, idx_before, idx_after in path:
        e_before = idx_to_e(idx_before) if h > 0 else E0
        if h == 0:
            e_before = E0
        e_after = idx_to_e(idx_after)
        delta = e_after - e_before
        charge_amt = max(0.0, delta)
        discharge_amt = max(0.0, -delta)

        d_h = demand[h]
        sol_h = effective_solar[h]

        needed = d_h + charge_amt - discharge_amt
        solar_used = min(sol_h, max(0.0, needed))
        grid = max(0.0, needed - solar_used)

        if charge_amt > EPS:
            action, mag = "charge", round(charge_amt, 4)
        elif discharge_amt > EPS:
            action, mag = "discharge", round(discharge_amt, 4)
        else:
            action, mag = "idle", 0.0

        plan.append(HourlyPlanEntry(
            hour=h,
            grid_kwh=round(grid, 4),
            solar_used_kwh=round(solar_used, 4),
            battery_action=action,
            battery_kwh=round(mag, 4),
            battery_energy_after_kwh=round(e_after, 4),
        ))

    return plan, total_cost, reachability


# ──────────────────────────────────────────────────────────────────────
# Phase 2: MILP solver with DP-derived bounds
# ──────────────────────────────────────────────────────────────────────

def _solve_milp(
    hours: List[HourEntry],
    battery: BatteryConfig,
    effective_solar: Dict[int, float],
    min_reserve: Dict[int, float],
    no_charge_hours: Set[int],
    no_discharge_hours: Set[int],
    max_grid: Dict[int, Optional[float]],
    reachability: Optional[List[Dict[str, float]]] = None,
    dp_plan: Optional[List[HourlyPlanEntry]] = None,
) -> Tuple[Optional[List[HourlyPlanEntry]], float]:
    """
    MILP solver using PuLP with binary variables for mutual exclusivity.
    Optionally uses DP-derived reachability bounds to tighten variable ranges.
    """
    n = 24
    demand = {h.hour: h.demand_kwh for h in hours}
    tariff = {h.hour: h.tariff_bdt_per_kwh for h in hours}

    prob = pulp.LpProblem("gridwise_hybrid_milp", pulp.LpMinimize)

    # ── Continuous variables ──
    grid = {h: pulp.LpVariable(f"grid_{h}", lowBound=0) for h in range(n)}
    solar_used = {
        h: pulp.LpVariable(f"solar_{h}", lowBound=0, upBound=effective_solar[h])
        for h in range(n)
    }

    # Charge/discharge with rate limits (no-charge/no-discharge enforced via upBound=0)
    charge = {
        h: pulp.LpVariable(
            f"charge_{h}", lowBound=0,
            upBound=0 if h in no_charge_hours else battery.max_charge_kwh_per_hour,
        )
        for h in range(n)
    }
    discharge = {
        h: pulp.LpVariable(
            f"discharge_{h}", lowBound=0,
            upBound=0 if h in no_discharge_hours else battery.max_discharge_kwh_per_hour,
        )
        for h in range(n)
    }

    # Battery energy with per-hour reserve and optional DP-tightened bounds
    energy = {}
    for h in range(n):
        lb = min_reserve[h]
        ub = battery.capacity_kwh
        if reachability is not None:
            # Tighten bounds from DP reachability (with small margin for fractional optima)
            margin = battery.max_charge_kwh_per_hour  # generous margin
            dp_lb = reachability[h]["min"] - margin
            dp_ub = reachability[h]["max"] + margin
            lb = max(lb, dp_lb)
            ub = min(ub, dp_ub)
            # Safety: don't over-tighten
            lb = max(lb, min_reserve[h])
            ub = min(ub, battery.capacity_kwh)
        energy[h] = pulp.LpVariable(f"energy_{h}", lowBound=lb, upBound=ub)

    # ── Binary variables for mutual exclusivity ──
    y_charge = {h: pulp.LpVariable(f"y_charge_{h}", cat="Binary") for h in range(n)}
    y_discharge = {h: pulp.LpVariable(f"y_discharge_{h}", cat="Binary") for h in range(n)}

    # ── Grid cap ──
    for h in range(n):
        if max_grid[h] is not None:
            grid[h].upBound = max_grid[h]

    # ── Objective: minimize total grid cost ──
    prob += pulp.lpSum(grid[h] * tariff[h] for h in range(n))

    # ── Constraints ──
    for h in range(n):
        # Energy balance
        prob += (
            grid[h] + solar_used[h] + discharge[h] == demand[h] + charge[h],
            f"balance_{h}",
        )

        # Battery state transition
        prev_energy = battery.initial_energy_kwh if h == 0 else energy[h - 1]
        prob += energy[h] == prev_energy + charge[h] - discharge[h], f"state_{h}"

        # Mutual exclusivity
        prob += y_charge[h] + y_discharge[h] <= 1, f"mutex_{h}"

        # Big-M linking: charge/discharge only when binary is 1
        if h not in no_charge_hours:
            prob += charge[h] <= battery.max_charge_kwh_per_hour * y_charge[h], f"link_charge_{h}"
        if h not in no_discharge_hours:
            prob += discharge[h] <= battery.max_discharge_kwh_per_hour * y_discharge[h], f"link_discharge_{h}"

    # End-of-day neutrality
    prob += energy[n - 1] == battery.initial_energy_kwh, "end_of_day_neutrality"

    # ── Warm start from DP solution ──
    if dp_plan is not None:
        plan_by_hour = {p.hour: p for p in dp_plan}
        for h in range(n):
            p = plan_by_hour.get(h)
            if p is None:
                continue
            grid[h].setInitialValue(p.grid_kwh)
            solar_used[h].setInitialValue(p.solar_used_kwh)
            energy[h].setInitialValue(p.battery_energy_after_kwh)
            if p.battery_action == "charge":
                charge[h].setInitialValue(p.battery_kwh)
                discharge[h].setInitialValue(0.0)
                y_charge[h].setInitialValue(1)
                y_discharge[h].setInitialValue(0)
            elif p.battery_action == "discharge":
                charge[h].setInitialValue(0.0)
                discharge[h].setInitialValue(p.battery_kwh)
                y_charge[h].setInitialValue(0)
                y_discharge[h].setInitialValue(1)
            else:
                charge[h].setInitialValue(0.0)
                discharge[h].setInitialValue(0.0)
                y_charge[h].setInitialValue(0)
                y_discharge[h].setInitialValue(0)

    # ── Solve ──
    solver = pulp.PULP_CBC_CMD(msg=0, warmStart=True if dp_plan else False)
    prob.solve(solver)

    if pulp.LpStatus[prob.status] != "Optimal":
        return None, float("inf")

    # ── Extract solution ──
    hourly_plan: List[HourlyPlanEntry] = []
    total_cost = 0.0

    for h in range(n):
        c = charge[h].value() or 0.0
        d = discharge[h].value() or 0.0

        # Net simultaneous charge+discharge (shouldn't happen with binaries, but guard)
        if c > EPS and d > EPS:
            if c >= d:
                c, d = c - d, 0.0
            else:
                c, d = 0.0, d - c

        if c > EPS:
            action, mag = "charge", c
        elif d > EPS:
            action, mag = "discharge", d
        else:
            action, mag = "idle", 0.0

        g = max(0.0, grid[h].value() or 0.0)
        s = max(0.0, solar_used[h].value() or 0.0)
        e = max(0.0, energy[h].value() or 0.0)
        total_cost += g * tariff[h]

        hourly_plan.append(HourlyPlanEntry(
            hour=h,
            grid_kwh=round(g, 4),
            solar_used_kwh=round(s, 4),
            battery_action=action,
            battery_kwh=round(max(0.0, mag), 4),
            battery_energy_after_kwh=round(e, 4),
        ))

    return hourly_plan, total_cost


# ──────────────────────────────────────────────────────────────────────
# Phase 3: Cross-validation + public solve() entry point
# ──────────────────────────────────────────────────────────────────────

def solve(
    hours: List[HourEntry],
    battery: BatteryConfig,
    directives: List[DirectiveInterpretation],
) -> List[HourlyPlanEntry]:
    """
    Hybrid MILP+DP solver — the main entry point called by main.py.

    Phase 1: DP scout (fast feasible + reachability bounds)
    Phase 2: MILP refine (exact optimal with warm start)
    Phase 3: Cross-validate and pick best
    """
    effective_solar, min_reserve, no_charge_hours, no_discharge_hours, max_grid = apply_directives(
        hours, battery, directives
    )

    # ── Phase 1: DP Scout ──
    logger.info("Phase 1: Running DP scout...")
    dp_plan, dp_cost, reachability = _solve_dp(
        hours, battery, effective_solar, min_reserve,
        no_charge_hours, no_discharge_hours, max_grid,
        granularity=0.5,
    )

    if dp_plan is None:
        logger.warning("DP found no feasible solution, trying MILP without warm start...")
        dp_cost = float("inf")

    logger.info("DP cost: %.2f BDT", dp_cost if dp_cost < float("inf") else -1)

    # ── Phase 2: MILP Refine ──
    logger.info("Phase 2: Running MILP with DP warm start...")
    milp_plan, milp_cost = _solve_milp(
        hours, battery, effective_solar, min_reserve,
        no_charge_hours, no_discharge_hours, max_grid,
        reachability=reachability if dp_plan else None,
        dp_plan=dp_plan,
    )

    logger.info("MILP cost: %.2f BDT", milp_cost if milp_cost < float("inf") else -1)

    # ── Phase 3: Cross-Validate ──
    if milp_plan is not None and milp_cost < float("inf"):
        if dp_plan is not None and dp_cost < float("inf"):
            gap = abs(dp_cost - milp_cost)
            logger.info("Cross-validation gap: %.4f BDT", gap)

            if gap > CROSS_VALIDATE_TOL:
                logger.warning(
                    "DP-MILP cost gap %.2f exceeds tolerance %.2f. DP=%.2f, MILP=%.2f",
                    gap, CROSS_VALIDATE_TOL, dp_cost, milp_cost,
                )

            if milp_cost > dp_cost + EPS:
                # MILP is worse than DP — unusual, may indicate MILP issue
                logger.warning("MILP cost > DP cost, using DP solution")
                return dp_plan

        return milp_plan

    # MILP failed — fall back to DP
    if dp_plan is not None:
        logger.warning("MILP failed, falling back to DP solution")
        return dp_plan

    raise OptimizationError("Both DP and MILP failed to find a feasible solution")
