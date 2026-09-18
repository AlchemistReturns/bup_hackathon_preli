import asyncio
import logging
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool

from app.config import REQUEST_TIMEOUT_SECONDS
from app.graph import InterpretationError, cache_source, run_interpretation
from app.optimizer import OptimizationError, solve
from app.replay import ReplayError, replay, replay_response
from app.schemas import HealthResponse, OptimizeEnergyResponse, ScenarioRequest

logger = logging.getLogger("gridwise")
app = FastAPI(title="GridWise LLM-Assisted Energy Optimizer")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Validation details may contain non-JSON NaN values or user data.
    return JSONResponse(status_code=400, content={"detail": "structurally invalid request"})


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok")


def _optimize(payload: ScenarioRequest):
    started = perf_counter()
    directives = run_interpretation(payload.operator_notes, payload.battery.capacity_kwh)
    interpreted = perf_counter()
    plan = solve(payload.hours, payload.battery, directives)
    solved = perf_counter()
    grid, cost, peak = replay(payload.hours, payload.battery, directives, plan)
    response = OptimizeEnergyResponse(
        scenario_id=payload.scenario_id,
        directive_interpretation=directives,
        hourly_plan=plan,
        total_grid_kwh=round(grid, 6),
        total_cost_bdt=round(cost, 6),
        peak_grid_kwh=round(peak, 6),
        plan_summary=(
            f"Applied {sum(d.applies for d in directives)} operator directive(s); "
            f"optimal grid cost {cost:.2f} BDT, peak grid draw {peak:.2f} kWh."
        ),
    )
    # Verify the actual JSON representation, not just raw solver arrays.
    response = OptimizeEnergyResponse.model_validate_json(response.model_dump_json())
    replay_response(payload, response)
    logger.info(
        "timings_ms interpretation=%.2f optimizer=%.2f validation=%.2f",
        (interpreted - started) * 1000, (solved - interpreted) * 1000,
        (perf_counter() - solved) * 1000,
    )
    timings = (
        f"interpretation;dur={(interpreted-started)*1000:.3f}, "
        f"optimizer;dur={(solved-interpreted)*1000:.3f}, "
        f"validation;dur={(perf_counter()-solved)*1000:.3f}"
    )
    return Response(
        content=response.model_dump_json(), media_type="application/json",
        headers={"Server-Timing": timings, "X-Interpretation-Cache": cache_source()},
    )


@app.post("/optimize-energy", response_model=OptimizeEnergyResponse)
async def optimize_energy(payload: ScenarioRequest):
    try:
        # Blocking HTTP/solver work leaves the event loop free for /health.
        return await asyncio.wait_for(
            run_in_threadpool(_optimize, payload), timeout=REQUEST_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        return JSONResponse(status_code=500, content={"detail": "request deadline exceeded"})
    except InterpretationError:
        return JSONResponse(status_code=500, content={"detail": "operator interpretation unavailable"})
    except (OptimizationError, ReplayError):
        return JSONResponse(status_code=500, content={"detail": "internal optimization error"})
    except Exception as exc:
        # Log the type only: SDK exception messages/tracebacks can expose data.
        logger.error("Request failed (%s)", type(exc).__name__)
        return JSONResponse(status_code=500, content={"detail": "internal server error"})
