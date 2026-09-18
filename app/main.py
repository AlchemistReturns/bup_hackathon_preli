import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import ValidationError

from app.schemas import HealthResponse, OptimizeEnergyResponse, ScenarioRequest
from app.graph import run_interpretation
from app.optimizer import solve, OptimizationError
from app.replay import replay, ReplayError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gridwise")

app = FastAPI(title="GridWise LLM-Assisted Energy Optimizer")


@app.exception_handler(ValidationError)
async def validation_exception_handler(request: Request, exc: ValidationError):
    return JSONResponse(status_code=400, content={"detail": "structurally invalid request", "errors": exc.errors()})


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok")


@app.post("/optimize-energy", response_model=OptimizeEnergyResponse)
async def optimize_energy(payload: ScenarioRequest):
    try:
        directives = run_interpretation(
            operator_notes=payload.operator_notes,
            battery_capacity=payload.battery.capacity_kwh,
        )

        hourly_plan = solve(payload.hours, payload.battery, directives)

        total_grid_kwh, total_cost_bdt, peak_grid_kwh = replay(
            payload.hours, payload.battery, directives, hourly_plan
        )

        applied = [d.directive_type for d in directives if d.applies]
        summary = (
            f"Applied {len(applied)} operator directive(s) ({', '.join(applied) or 'none'}); "
            f"total grid cost {total_cost_bdt:.2f} BDT, peak grid draw {peak_grid_kwh:.2f} kWh."
        )

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
