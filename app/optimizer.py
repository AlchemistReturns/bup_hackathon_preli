"""
Deterministic LP optimizer. Operator directives are applied here as
constraints/parameters — never trusted math from the LLM, only validated
structured_adjustment objects that guardrails.py has already accepted.
"""
from typing import Dict, List
import pulp

from app.schemas import BatteryConfig, DirectiveInterpretation, HourEntry, HourlyPlanEntry

EPS = 1e-6


class OptimizationError(Exception):
    pass


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


def solve(
    hours: List[HourEntry],
    battery: BatteryConfig,
    directives: List[DirectiveInterpretation],
):
    n = 24
    demand = {h.hour: h.demand_kwh for h in hours}
    tariff = {h.hour: h.tariff_bdt_per_kwh for h in hours}
    effective_solar, min_reserve, no_charge_hours, no_discharge_hours, max_grid = apply_directives(
        hours, battery, directives
    )

    prob = pulp.LpProblem("gridwise_energy", pulp.LpMinimize)

    grid = {h: pulp.LpVariable(f"grid_{h}", lowBound=0) for h in range(n)}
    solar_used = {h: pulp.LpVariable(f"solar_used_{h}", lowBound=0, upBound=effective_solar[h]) for h in range(n)}
    charge = {
        h: pulp.LpVariable(f"charge_{h}", lowBound=0, upBound=0 if h in no_charge_hours else battery.max_charge_kwh_per_hour)
        for h in range(n)
    }
    discharge = {
        h: pulp.LpVariable(
            f"discharge_{h}", lowBound=0, upBound=0 if h in no_discharge_hours else battery.max_discharge_kwh_per_hour
        )
        for h in range(n)
    }
    energy = {h: pulp.LpVariable(f"energy_{h}", lowBound=min_reserve[h], upBound=battery.capacity_kwh) for h in range(n)}

    if max_grid[0] is not None:
        grid[0].upBound = max_grid[0]
    for h in range(n):
        if max_grid[h] is not None:
            grid[h].upBound = max_grid[h]

    prob += pulp.lpSum(grid[h] * tariff[h] for h in range(n))

    for h in range(n):
        prob += grid[h] + solar_used[h] + discharge[h] == demand[h] + charge[h], f"balance_{h}"
        prev_energy = battery.initial_energy_kwh if h == 0 else energy[h - 1]
        prob += energy[h] == prev_energy + charge[h] - discharge[h], f"state_{h}"

    prob += energy[n - 1] == battery.initial_energy_kwh, "end_of_day_neutrality"

    solver = pulp.PULP_CBC_CMD(msg=0)
    prob.solve(solver)

    if pulp.LpStatus[prob.status] != "Optimal":
        raise OptimizationError(f"No optimal solution found (status={pulp.LpStatus[prob.status]})")

    hourly_plan: List[HourlyPlanEntry] = []
    for h in range(n):
        c = charge[h].value() or 0.0
        d = discharge[h].value() or 0.0
        if c > EPS and d > EPS:
            # Shouldn't happen for a cost-minimal solution, but guard anyway.
            if c >= d:
                d = 0.0
            else:
                c = 0.0
        if c > EPS:
            action, mag = "charge", c
        elif d > EPS:
            action, mag = "discharge", d
        else:
            action, mag = "idle", 0.0

        hourly_plan.append(
            HourlyPlanEntry(
                hour=h,
                grid_kwh=max(0.0, grid[h].value() or 0.0),
                solar_used_kwh=max(0.0, solar_used[h].value() or 0.0),
                battery_action=action,
                battery_kwh=max(0.0, mag),
                battery_energy_after_kwh=max(0.0, energy[h].value() or 0.0),
            )
        )

    return hourly_plan
