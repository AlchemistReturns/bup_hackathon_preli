"""
LangGraph pipeline for operator-note interpretation.

    call_llm -> validate -> (ok: END) | (bad, retries left: call_llm) | (bad, no retries: fallback -> END)

The LLM's output is never trusted directly: `validate` runs the deterministic
guardrails in app/guardrails.py. On repeated guardrail failure the pipeline
fails safe by marking every unresolved note as no_op rather than inventing
an energy rule or crashing (per Section 08, SAFE FAILURE).
"""
from typing import List, Optional, TypedDict
from langgraph.graph import StateGraph, END

from app.config import LLM_MAX_RETRIES
from app.guardrails import validate_all, GuardrailError
from app.llm_client import interpret_notes
from app.schemas import DirectiveInterpretation


class PipelineState(TypedDict):
    operator_notes: List[str]
    battery_capacity: float
    raw_output: Optional[list]
    error: Optional[str]
    attempts: int
    validated: Optional[List[DirectiveInterpretation]]
    safe_failed: bool


def node_call_llm(state: PipelineState) -> PipelineState:
    state["attempts"] += 1
    try:
        raw = interpret_notes(
            state["operator_notes"],
            feedback=state.get("error"),
            battery_capacity=state.get("battery_capacity"),
        )
        state["raw_output"] = raw
        state["error"] = None
    except Exception as e:
        state["raw_output"] = None
        state["error"] = str(e)
    return state


def node_validate(state: PipelineState) -> PipelineState:
    if state["raw_output"] is None:
        # LLM call itself failed (e.g. bad JSON); error already set.
        return state
    try:
        validated = validate_all(
            state["raw_output"],
            num_notes=len(state["operator_notes"]),
            battery_capacity=state["battery_capacity"],
            operator_notes=state.get("operator_notes"),
        )
        state["validated"] = validated
        state["error"] = None
    except GuardrailError as e:
        state["error"] = str(e)
    return state


def node_fallback(state: PipelineState) -> PipelineState:
    """Safe failure: mark every note as no_op instead of crashing or guessing."""
    state["validated"] = [
        DirectiveInterpretation(
            note_index=i,
            applies=False,
            directive_type="no_op",
            structured_adjustment=None,
            explanation="Safe fallback: could not obtain a valid interpretation for this note.",
        )
        for i in range(len(state["operator_notes"]))
    ]
    state["safe_failed"] = True
    return state


def route_after_validate(state: PipelineState) -> str:
    if state.get("error") is None and state.get("validated") is not None:
        return "done"
    if state["attempts"] <= LLM_MAX_RETRIES:
        return "retry"
    return "fallback"


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("call_llm", node_call_llm)
    graph.add_node("validate", node_validate)
    graph.add_node("fallback", node_fallback)

    graph.set_entry_point("call_llm")
    graph.add_edge("call_llm", "validate")
    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {"done": END, "retry": "call_llm", "fallback": "fallback"},
    )
    graph.add_edge("fallback", END)
    return graph.compile()


_compiled_graph = None


def get_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph


def run_interpretation(operator_notes: List[str], battery_capacity: float) -> List[DirectiveInterpretation]:
    initial_state: PipelineState = {
        "operator_notes": operator_notes,
        "battery_capacity": battery_capacity,
        "raw_output": None,
        "error": None,
        "attempts": 0,
        "validated": None,
        "safe_failed": False,
    }
    final_state = get_graph().invoke(initial_state)
    return final_state["validated"]
