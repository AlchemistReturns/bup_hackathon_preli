"""
Independent replay of the final schedule (Section 11.3 / 9). Runs server-side
as a sanity check before the response is returned, mirroring what the hidden
judge will do.
"""
from typing import List, Tuple
from app.schemas import BatteryConfig, DirectiveInterpretation, HourEntry, HourlyPlanEntry
from app.optimizer import apply_directives

TOL = 0.01


class ReplayError(Exception):
    pass


def replay(
    hours: List[HourEntry],
    battery: BatteryConfig,
    directives: List[DirectiveInterpretation],
    plan: List[HourlyPlanEntry],
) -> Tuple[float, float, float]:
    if len(plan) != 24 or sorted(p.hour for p in plan) != list(range(24)):
        raise ReplayError("hourly_plan must contain exactly 24 unique hours 0-23")

    demand = {h.hour: h.demand_kwh for h in hours}
    tariff = {h.hour: h.tariff_bdt_per_kwh for h in hours}
    effective_solar, min_reserve, no_charge_hours, no_discharge_hours, max_grid = apply_directives(
        hours, battery, directives
    )

    plan_by_hour = {p.hour: p for p in plan}
    prev_energy = battery.initial_energy_kwh
    total_grid = 0.0
    total_cost = 0.0
    peak_grid = 0.0

    for h in range(24):
        p = plan_by_hour[h]

        if p.solar_used_kwh > effective_solar[h] + TOL:
            raise ReplayError(f"hour {h}: solar_used_kwh exceeds effective available solar")

        if p.battery_action == "charge":
            if h in no_charge_hours and p.battery_kwh > TOL:
                raise ReplayError(f"hour {h}: charging is disallowed by a no_charge_window directive")
            if p.battery_kwh > battery.max_charge_kwh_per_hour + TOL:
                raise ReplayError(f"hour {h}: charge exceeds max_charge_kwh_per_hour")
            energy_after = prev_energy + p.battery_kwh
        elif p.battery_action == "discharge":
            if h in no_discharge_hours and p.battery_kwh > TOL:
                raise ReplayError(f"hour {h}: discharging is disallowed by a no_discharge_window directive")
            if p.battery_kwh > battery.max_discharge_kwh_per_hour + TOL:
                raise ReplayError(f"hour {h}: discharge exceeds max_discharge_kwh_per_hour")
            energy_after = prev_energy - p.battery_kwh
        else:
            energy_after = prev_energy

        if abs(energy_after - p.battery_energy_after_kwh) > TOL:
            raise ReplayError(f"hour {h}: battery_energy_after_kwh does not match battery action")

        if energy_after < min_reserve[h] - TOL or energy_after > battery.capacity_kwh + TOL:
            raise ReplayError(f"hour {h}: battery_energy_after_kwh out of bounds")

        if max_grid[h] is not None and p.grid_kwh > max_grid[h] + TOL:
            raise ReplayError(f"hour {h}: grid_kwh exceeds max_grid_window limit")

        lhs = p.grid_kwh + p.solar_used_kwh + (p.battery_kwh if p.battery_action == "discharge" else 0.0)
        rhs = demand[h] + (p.battery_kwh if p.battery_action == "charge" else 0.0)
        if abs(lhs - rhs) > TOL:
            raise ReplayError(f"hour {h}: energy balance does not hold")

        total_grid += p.grid_kwh
        total_cost += p.grid_kwh * tariff[h]
        peak_grid = max(peak_grid, p.grid_kwh)
        prev_energy = energy_after

    if abs(prev_energy - battery.initial_energy_kwh) > TOL:
        raise ReplayError("final battery_energy_after_kwh does not equal initial_energy_kwh")

    return total_grid, total_cost, peak_grid
