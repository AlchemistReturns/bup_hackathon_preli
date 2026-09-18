"""Shared builders and an independent physics checker (deliberately does not use app.replay)."""
from pathlib import Path

from app.schemas import BatteryConfig, DirectiveInterpretation, HourEntry

ROOT = Path(__file__).resolve().parent.parent
CASES_FILE = "BUP_CSE_FEST_2026_Preli_Public_Sample_Cases.json"


def find_cases_file() -> Path:
    for p in (ROOT / "temporary" / CASES_FILE, ROOT / CASES_FILE, ROOT / "tests" / "data" / CASES_FILE):
        if p.exists():
            return p
    raise FileNotFoundError(f"{CASES_FILE} not found under temporary/, repo root, or tests/data/")


def make_hours(demand, solar, tariff):
    return [
        HourEntry(hour=h, demand_kwh=demand[h], solar_kwh=solar[h], tariff_bdt_per_kwh=tariff[h])
        for h in range(24)
    ]


def make_battery(cap=200, init=100, min_=40, chg=50, dis=50):
    return BatteryConfig(
        capacity_kwh=cap, initial_energy_kwh=init, minimum_energy_kwh=min_,
        max_charge_kwh_per_hour=chg, max_discharge_kwh_per_hour=dis,
    )


def directive(dtype, **adj):
    return DirectiveInterpretation(
        note_index=0, applies=True, directive_type=dtype, structured_adjustment=adj, explanation="t"
    )


def check_physics(hours, battery, eff_solar, plan, tol=1e-5):
    """Return a list of physics violations (empty == valid). Directive limits other than the
    effective-solar ceiling are NOT checked here."""
    errs = []
    if sorted(p.hour for p in plan) != list(range(24)):
        return ["plan does not cover hours 0-23 exactly once"]
    demand = {h.hour: h.demand_kwh for h in hours}
    by_h = {p.hour: p for p in plan}
    prev = battery.initial_energy_kwh
    for h in range(24):
        p = by_h[h]
        if p.grid_kwh < 0 or p.solar_used_kwh < 0:
            errs.append(f"h{h}: negative grid/solar")
        if p.solar_used_kwh > eff_solar[h] + tol:
            errs.append(f"h{h}: solar_used {p.solar_used_kwh} > effective {eff_solar[h]}")
        if p.battery_action == "idle":
            if p.battery_kwh != 0:
                errs.append(f"h{h}: idle with battery_kwh={p.battery_kwh}")
            delta = 0.0
        elif p.battery_action == "charge":
            if p.battery_kwh > battery.max_charge_kwh_per_hour + tol:
                errs.append(f"h{h}: charge {p.battery_kwh} > max")
            delta = p.battery_kwh
        else:
            if p.battery_kwh > battery.max_discharge_kwh_per_hour + tol:
                errs.append(f"h{h}: discharge {p.battery_kwh} > max")
            delta = -p.battery_kwh
        lhs = p.grid_kwh + p.solar_used_kwh + (p.battery_kwh if p.battery_action == "discharge" else 0.0)
        rhs = demand[h] + (p.battery_kwh if p.battery_action == "charge" else 0.0)
        if abs(lhs - rhs) > tol:
            errs.append(f"h{h}: balance {lhs} != {rhs}")
        if abs(prev + delta - p.battery_energy_after_kwh) > tol:
            errs.append(f"h{h}: energy chain {prev}+{delta} != {p.battery_energy_after_kwh}")
        if not (battery.minimum_energy_kwh - tol <= p.battery_energy_after_kwh <= battery.capacity_kwh + tol):
            errs.append(f"h{h}: energy {p.battery_energy_after_kwh} outside [base_min, capacity]")
        prev = p.battery_energy_after_kwh
    if abs(prev - battery.initial_energy_kwh) > 1e-9:
        errs.append(f"end-of-day energy {prev} != initial {battery.initial_energy_kwh}")
    return errs
