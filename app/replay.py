"""Independent physical replay, without using the optimizer's compiler."""
import math
from typing import List, Tuple

from app.schemas import BatteryConfig, DirectiveInterpretation, HourEntry, HourlyPlanEntry

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
    by_hour = {h.hour: h for h in hours}
    plan_by_hour = {p.hour: p for p in plan}
    energy = battery.initial_energy_kwh
    grid_values, costs = [], []
    for h in range(24):
        row, source = plan_by_hour[h], by_hour[h]
        values = [row.grid_kwh, row.solar_used_kwh, row.battery_kwh, row.battery_energy_after_kwh]
        if any(not math.isfinite(v) or v < 0 for v in values):
            raise ReplayError(f"hour {h}: non-finite or negative energy")
        if row.battery_action not in ("charge", "discharge", "idle"):
            raise ReplayError(f"hour {h}: invalid action")
        if row.battery_action == "idle" and row.battery_kwh != 0:
            raise ReplayError(f"hour {h}: idle battery must have zero movement")

        # Re-evaluate every directive directly against the emitted action.
        solar_limit = source.solar_kwh
        reserve = battery.minimum_energy_kwh
        for directive in directives:
            if not directive.applies:
                continue
            adjustment = directive.structured_adjustment
            if h not in adjustment["hours"]:
                continue
            kind = directive.directive_type
            if kind == "solar_reduction":
                solar_limit *= adjustment["factor"]
            elif kind == "minimum_battery_reserve":
                reserve = max(reserve, adjustment["minimum_energy_kwh"])
            elif kind == "max_grid_window":
                if row.grid_kwh > adjustment["max_grid_kwh"] + TOL:
                    raise ReplayError(f"hour {h}: grid cap violated")
            elif kind == "no_charge_window":
                if row.battery_action == "charge" and row.battery_kwh > TOL:
                    raise ReplayError(f"hour {h}: charging prohibited")
            elif kind == "no_discharge_window":
                if row.battery_action == "discharge" and row.battery_kwh > TOL:
                    raise ReplayError(f"hour {h}: discharging prohibited")
            else:
                raise ReplayError("Unknown directive")

        movement = 0.0
        if row.battery_action == "charge":
            if row.battery_kwh > battery.max_charge_kwh_per_hour + TOL:
                raise ReplayError(f"hour {h}: charge limit violated")
            movement = row.battery_kwh
        elif row.battery_action == "discharge":
            if row.battery_kwh > battery.max_discharge_kwh_per_hour + TOL:
                raise ReplayError(f"hour {h}: discharge limit violated")
            movement = -row.battery_kwh
        energy += movement
        if abs(energy - row.battery_energy_after_kwh) > TOL:
            raise ReplayError(f"hour {h}: battery transition mismatch")
        if energy < reserve - TOL or energy > battery.capacity_kwh + TOL:
            raise ReplayError(f"hour {h}: reserve/capacity violated")
        if row.solar_used_kwh > solar_limit + TOL:
            raise ReplayError(f"hour {h}: unavailable solar used")
        if abs(row.grid_kwh + row.solar_used_kwh - source.demand_kwh - movement) > TOL:
            raise ReplayError(f"hour {h}: energy balance failed")
        grid_values.append(row.grid_kwh)
        costs.append(row.grid_kwh * source.tariff_bdt_per_kwh)

    if abs(energy - battery.initial_energy_kwh) > TOL:
        raise ReplayError("End-of-day neutrality failed")
    totals = math.fsum(grid_values), math.fsum(costs), max(grid_values)
    if not all(math.isfinite(v) for v in totals):
        raise ReplayError("Non-finite totals")
    return totals


def replay_response(request, response):
    if response.scenario_id != request.scenario_id:
        raise ReplayError("Scenario id mismatch")
    from app.guardrails import validate_all
    validated = validate_all(
        [d.model_dump() for d in response.directive_interpretation],
        len(request.operator_notes), request.battery.capacity_kwh,
    )
    totals = replay(request.hours, request.battery, validated, response.hourly_plan)
    reported = response.total_grid_kwh, response.total_cost_bdt, response.peak_grid_kwh
    if any(not math.isfinite(actual) or abs(actual - expected) > TOL
           for actual, expected in zip(reported, totals)):
        raise ReplayError("Reported totals disagree with the emitted plan")
