"""Exact 48-variable LP: grid[0:24], battery energy[24:48].

delta[h] = energy[h] - energy[h-1], using initial energy at hour zero.
solar[h] = demand[h] + delta[h] - grid[h]. Eliminating the other variables
preserves the challenge's lossless battery model exactly.
"""
from typing import List

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from app.config import SOLVER_TIMEOUT_SECONDS
from app.schemas import BatteryConfig, DirectiveInterpretation, HourEntry, HourlyPlanEntry

N = 24
EPS = 1e-7

# Build the sparse pattern once. Only request-local bounds, costs and RHS vary.
# No shared mutable solver state: concurrent requests cannot overwrite each other.
_IDENTITY = sparse.eye(N, format="csr")
_DELTA = sparse.diags([np.ones(N), -np.ones(N - 1)], [0, -1], format="csr")
_ZERO = sparse.csr_matrix((N, N))
_A_UB = sparse.vstack([
    sparse.hstack([-_IDENTITY, _DELTA]),  # solar <= available
    sparse.hstack([_IDENTITY, -_DELTA]),  # solar >= 0
    sparse.hstack([_ZERO, _DELTA]),      # delta <= charge limit
    sparse.hstack([_ZERO, -_DELTA]),     # delta >= -discharge limit
], format="csr")
_A_EQ = sparse.csr_matrix(([1.0], ([0], [2 * N - 1])), shape=(1, 2 * N))


class OptimizationError(Exception):
    pass


def apply_directives(hours, battery, directives):
    """Compile validated directives; replay intentionally does not use this."""
    solar = {h.hour: h.solar_kwh for h in hours}
    reserve = {h: battery.minimum_energy_kwh for h in range(N)}
    no_charge, no_discharge = set(), set()
    grid_cap = {h: None for h in range(N)}
    for directive in directives:
        if not directive.applies:
            continue
        adjustment = directive.structured_adjustment
        for h in adjustment["hours"]:
            kind = directive.directive_type
            if kind == "solar_reduction":
                solar[h] *= adjustment["factor"]
            elif kind == "minimum_battery_reserve":
                reserve[h] = max(reserve[h], adjustment["minimum_energy_kwh"])
            elif kind == "no_charge_window":
                no_charge.add(h)
            elif kind == "no_discharge_window":
                no_discharge.add(h)
            elif kind == "max_grid_window":
                cap = adjustment["max_grid_kwh"]
                grid_cap[h] = cap if grid_cap[h] is None else min(grid_cap[h], cap)
            else:
                raise OptimizationError("Unsupported directive")
    return solar, reserve, no_charge, no_discharge, grid_cap


def solve(
    hours: List[HourEntry],
    battery: BatteryConfig,
    directives: List[DirectiveInterpretation],
) -> List[HourlyPlanEntry]:
    by_hour = {entry.hour: entry for entry in hours}
    if len(hours) != N or set(by_hour) != set(range(N)):
        raise OptimizationError("Expected exactly hours 0 through 23")
    solar, reserve, no_charge, no_discharge, caps = apply_directives(hours, battery, directives)
    demand = np.array([by_hour[h].demand_kwh for h in range(N)])
    available = np.array([solar[h] for h in range(N)])
    charge_limit = np.array([0 if h in no_charge else battery.max_charge_kwh_per_hour for h in range(N)])
    discharge_limit = np.array([0 if h in no_discharge else battery.max_discharge_kwh_per_hour for h in range(N)])
    initial_offset = np.zeros(N)
    initial_offset[0] = battery.initial_energy_kwh
    b_ub = np.concatenate([
        available - demand + initial_offset,
        demand - initial_offset,
        charge_limit + initial_offset,
        discharge_limit - initial_offset,
    ])
    costs = np.array([by_hour[h].tariff_bdt_per_kwh for h in range(N)] + [0.0] * N)
    bounds = [(0.0, caps[h]) for h in range(N)] + [(reserve[h], battery.capacity_kwh) for h in range(N)]
    try:
        result = linprog(
            costs, A_ub=_A_UB, b_ub=b_ub,
            A_eq=_A_EQ, b_eq=[battery.initial_energy_kwh],
            bounds=bounds, method="highs",
            options={
                "time_limit": SOLVER_TIMEOUT_SECONDS,
                "primal_feasibility_tolerance": 1e-9,
                "dual_feasibility_tolerance": 1e-9,
            },
        )
    except Exception as exc:
        raise OptimizationError("Solver execution failed") from exc
    if not result.success or result.x is None or not np.all(np.isfinite(result.x)):
        raise OptimizationError("No finite optimal solution found")

    plan = []
    previous = battery.initial_energy_kwh
    for h in range(N):
        energy = float(result.x[N + h])
        movement = energy - previous
        if abs(movement) <= 1e-9:
            movement, energy = 0.0, previous
        grid = float(result.x[h])
        solar_used = float(demand[h] + movement - grid)
        if grid < -EPS or solar_used < -EPS or energy < -EPS:
            raise OptimizationError("Solver returned an invalid energy value")
        plan.append(HourlyPlanEntry(
            hour=h,
            grid_kwh=max(0.0, grid),
            solar_used_kwh=max(0.0, solar_used),
            battery_action="charge" if movement > 0 else "discharge" if movement < 0 else "idle",
            battery_kwh=abs(movement),
            battery_energy_after_kwh=max(0.0, energy),
        ))
        previous = energy
    return plan
