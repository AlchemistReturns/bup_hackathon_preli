import logging
import math
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.schemas import DirectiveInterpretation, HealthResponse, OptimizeEnergyResponse, ScenarioRequest
from app.graph import run_interpretation
from app.optimizer import solve, trivial_plan, OptimizationError
from app.replay import replay, ReplayError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gridwise")

app = FastAPI(title="GridWise LLM-Assisted Energy Optimizer")


def _plan_and_validate(payload: ScenarioRequest, directives):
    """Solve, then replay-validate. Returns (plan, mode, (total_grid, total_cost, peak_grid)).

    mode: "optimal" | "relaxed" (directives infeasible, minimum-violation plan) | "fallback" (grid-only plan).
    replay() also enforces directive limits, which a relaxed plan cannot meet by definition, so relaxed and
    fallback plans are validated for physics only (replay with no directives).
    """
    plan, mode = None, "fallback"
    try:
        plan, mode = solve(payload.hours, payload.battery, directives)
        return plan, mode, replay(payload.hours, payload.battery, directives, plan)
    except (OptimizationError, ReplayError) as e:
        logger.warning("scenario %s: %s (mode=%s)", payload.scenario_id, e, mode)

    if plan is not None and mode == "relaxed":
        try:
            return plan, mode, replay(payload.hours, payload.battery, [], plan)
        except ReplayError as e:
            logger.error("scenario %s: relaxed plan failed physics replay: %s", payload.scenario_id, e)

    plan = trivial_plan(payload.hours, payload.battery)
    try:
        totals = replay(payload.hours, payload.battery, directives, plan)
    except ReplayError:  # trivial plan is physics-valid but may breach a directive limit
        totals = replay(payload.hours, payload.battery, [], plan)
    return plan, "fallback", totals


@app.exception_handler(RequestValidationError)
async def request_validation_handler(request: Request, exc: RequestValidationError):
    # FastAPI's default is 422; the spec wants 400 for malformed JSON / structurally invalid requests.
    errors = [{"loc": list(e.get("loc", [])), "msg": e.get("msg"), "type": e.get("type")} for e in exc.errors()]
    return JSONResponse(status_code=400, content={"detail": "structurally invalid request", "errors": errors})


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(status_code=400, content={"detail": "structurally invalid request", "errors": exc.errors()})


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok")


def _non_finite_fields(payload: ScenarioRequest) -> list:
    """Names of numeric request fields that are NaN/Infinity. Pydantic's ge=0 lets +Infinity through, but
    Section 11.3 requires finite numbers and an infinite value would poison the LP."""
    bad = []
    for h in payload.hours:
        for f in ("demand_kwh", "solar_kwh", "tariff_bdt_per_kwh"):
            if not math.isfinite(getattr(h, f)):
                bad.append(f"hours[{h.hour}].{f}")
    for f in ("capacity_kwh", "initial_energy_kwh", "minimum_energy_kwh",
              "max_charge_kwh_per_hour", "max_discharge_kwh_per_hour"):
        if not math.isfinite(getattr(payload.battery, f)):
            bad.append(f"battery.{f}")
    return bad


def _all_no_op(n: int, why: str):
    return [
        DirectiveInterpretation(note_index=i, applies=False, directive_type="no_op",
                                structured_adjustment=None, explanation=why)
        for i in range(n)
    ]


def _interpret(payload: ScenarioRequest):
    """run_interpretation() with a last-resort containment: anything that escapes the LangGraph pipeline
    (including non-GuardrailError crashes inside the guardrails) becomes the Section 08 safe failure
    (every note no_op) instead of an HTTP 500."""
    n = len(payload.operator_notes)
    try:
        out = run_interpretation(operator_notes=payload.operator_notes, battery_capacity=payload.battery.capacity_kwh)
        if not isinstance(out, list) or len(out) != n:
            raise ValueError(f"interpretation returned {type(out).__name__} of wrong length")
        return out
    except Exception:
        logger.exception("scenario %s: interpretation failed unexpectedly; using safe no_op fallback", payload.scenario_id)
        return _all_no_op(n, "Safe fallback: could not obtain a valid interpretation for this note.")


@app.post("/optimize-energy", response_model=OptimizeEnergyResponse)
async def optimize_energy(payload: ScenarioRequest):
    bad = _non_finite_fields(payload)
    if bad:
        return JSONResponse(status_code=400, content={"detail": "structurally invalid request",
                                                      "errors": [{"loc": ["body", f], "msg": "must be finite"} for f in bad]})
    try:
        directives = _interpret(payload)

        hourly_plan, mode, (total_grid_kwh, total_cost_bdt, peak_grid_kwh) = _plan_and_validate(
            payload, directives
        )
        logger.info("scenario %s solved, mode=%s", payload.scenario_id, mode)

        applied = [d.directive_type for d in directives if d.applies]
        summary = (
            f"Applied {len(applied)} operator directive(s) ({', '.join(applied) or 'none'}); "
            f"total grid cost {total_cost_bdt:.2f} BDT, peak grid draw {peak_grid_kwh:.2f} kWh."
        )
        if mode == "relaxed":
            summary += " Directives were mutually infeasible; returned the minimum-violation plan."
        elif mode == "fallback":
            summary += " Optimizer output failed validation; returned a grid-only fallback plan."

        return OptimizeEnergyResponse(
            scenario_id=payload.scenario_id,
            directive_interpretation=directives,
            hourly_plan=hourly_plan,
            total_grid_kwh=round(total_grid_kwh, 2),
            total_cost_bdt=round(total_cost_bdt, 2),
            peak_grid_kwh=round(peak_grid_kwh, 2),
            plan_summary=summary,
        )

    except (OptimizationError, ReplayError) as e:
        logger.error("scenario %s failed: %s", payload.scenario_id, e)
        return JSONResponse(status_code=500, content={"detail": "internal optimization error"})
    except Exception as e:
        logger.exception("unexpected error for scenario %s", payload.scenario_id)
        return JSONResponse(status_code=500, content={"detail": "internal server error"})
