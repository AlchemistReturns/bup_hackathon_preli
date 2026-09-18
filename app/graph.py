"""Bounded LLM -> validation pipeline. Failure never becomes a no_op."""
from functools import lru_cache
from contextvars import ContextVar
from time import monotonic
from typing import List, Optional, TypedDict

from langgraph.graph import END, StateGraph

from app.config import LLM_BUDGET_SECONDS, LLM_MAX_RETRIES, LLM_TIMEOUT_SECONDS
from app.config import LLM_CACHE_SIZE, LLM_CACHE_TTL_SECONDS, OPENAI_MODEL, OPENAI_API_MODE
from app.interpretation_cache import InterpretationCache
from app.llm_protocol import PROTOCOL_VERSION
from app.guardrails import GuardrailError, validate_all
from app.llm_client import interpret_notes
from app.schemas import DirectiveInterpretation

_cache = InterpretationCache(LLM_CACHE_SIZE, LLM_CACHE_TTL_SECONDS)
_cache_source = ContextVar("interpretation_cache_source", default="miss")


def cache_source():
    return _cache_source.get()


class InterpretationError(Exception):
    pass


class PipelineState(TypedDict):
    operator_notes: List[str]
    battery_capacity: float
    raw_output: Optional[list]
    error: Optional[str]
    attempts: int
    validated: Optional[List[DirectiveInterpretation]]
    deadline: float


def node_call_llm(state: PipelineState) -> PipelineState:
    state["attempts"] += 1
    state["validated"] = None
    remaining = state["deadline"] - monotonic()
    if remaining <= 0:
        raise InterpretationError("Interpretation deadline exceeded")
    try:
        state["raw_output"] = interpret_notes(
            state["operator_notes"],
            battery_capacity=state["battery_capacity"],
            feedback=state.get("error"),
            timeout=min(LLM_TIMEOUT_SECONDS, remaining),
        )
        state["error"] = None
    except GuardrailError as exc:
        state["raw_output"] = None
        state["error"] = str(exc)
    except Exception:
        state["raw_output"] = None
        # Provider exceptions may contain raw request data. Never echo them.
        state["error"] = "Model call failed or returned invalid JSON."
    return state


def node_validate(state: PipelineState) -> PipelineState:
    if state["raw_output"] is None:
        return state
    try:
        state["validated"] = validate_all(
            state["raw_output"],
            num_notes=len(state["operator_notes"]),
            battery_capacity=state["battery_capacity"],
        )
        state["error"] = None
    except GuardrailError as exc:
        state["error"] = str(exc)
    return state


def node_failure(state: PipelineState) -> PipelineState:
    raise InterpretationError("Could not obtain validated operator directives")


def route_after_validate(state: PipelineState) -> str:
    if monotonic() >= state["deadline"]:
        return "failure"
    if state.get("error") is None and state.get("validated") is not None:
        return "done"
    if state["attempts"] <= LLM_MAX_RETRIES:
        return "retry"
    return "failure"


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("call_llm", node_call_llm)
    graph.add_node("validate", node_validate)
    graph.add_node("failure", node_failure)
    graph.set_entry_point("call_llm")
    graph.add_edge("call_llm", "validate")
    graph.add_conditional_edges(
        "validate", route_after_validate,
        {"done": END, "retry": "call_llm", "failure": "failure"},
    )
    graph.add_edge("failure", END)
    return graph.compile()


@lru_cache(maxsize=1)
def get_graph():
    return build_graph()


def _run_uncached(operator_notes: List[str], battery_capacity: float) -> List[DirectiveInterpretation]:
    state: PipelineState = {
        "operator_notes": operator_notes,
        "battery_capacity": battery_capacity,
        "raw_output": None,
        "error": None,
        "attempts": 0,
        "validated": None,
        "deadline": monotonic() + LLM_BUDGET_SECONDS,
    }
    result = get_graph().invoke(state)
    if result.get("validated") is None:
        raise InterpretationError("No validated interpretation")
    return result["validated"]


def run_interpretation(operator_notes: List[str], battery_capacity: float) -> List[DirectiveInterpretation]:
    # Exact text and order only. Capacity matters for percentage reserves.
    # Scenario IDs and energy forecasts do not affect interpretation; the full
    # optimization and independent replay still run on every request.
    key = (PROTOCOL_VERSION, OPENAI_API_MODE, OPENAI_MODEL, battery_capacity, tuple(operator_notes))
    _cache_source.set("miss")
    try:
        result, source = _cache.get_or_compute(
            key, lambda: _run_uncached(operator_notes, battery_capacity),
            timeout=LLM_BUDGET_SECONDS,
        )
    except TimeoutError as exc:
        raise InterpretationError("Interpretation deadline exceeded") from exc
    _cache_source.set(source)
    return result
