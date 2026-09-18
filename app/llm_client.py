import json
from typing import List, Optional
from openai import OpenAI
from app.config import OPENAI_API_KEY, OPENAI_MODEL

_client: Optional[OpenAI] = None


def get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    return _client


SYSTEM_PROMPT = """You are the operator-note interpreter for a campus energy optimizer.

Convert each operator note into exactly one directive. Supported directive types:

- solar_reduction: {"hours": [...], "factor": number}   // factor = fraction of solar that REMAINS (80% reduction -> factor 0.2)
- minimum_battery_reserve: {"hours": [...], "minimum_energy_kwh": number}
- no_charge_window: {"hours": [...]}
- no_discharge_window: {"hours": [...]}
- max_grid_window: {"hours": [...], "max_grid_kwh": number}
- no_op: structured_adjustment must be null (use for notes that do not change the 24-hour energy schedule)

Rules:
- Time windows are whole hours. "1 PM to 3 PM" means hours [13, 14] (end hour excluded).
- hours must be unique integers 0-23 in ascending order.
- Do not invent demand, solar, tariff, or battery parameters.
- Do not invent directive types outside the six listed above.
- Return applies=true for every directive except no_op, which must have applies=false.
- Output ONLY a JSON array, one object per note, in the same order as the notes, with fields:
  note_index, applies, directive_type, structured_adjustment, explanation.
No prose, no markdown fences, JSON array only.
"""


def build_user_prompt(operator_notes: List[str]) -> str:
    lines = [f"{i}: {note}" for i, note in enumerate(operator_notes)]
    return "Operator notes:\n" + "\n".join(lines)


def interpret_notes(operator_notes: List[str], feedback: Optional[str] = None) -> list:
    """Calls the OpenAI Responses API and returns the parsed raw JSON list.
    Raises ValueError if the response is not valid JSON.
    """
    client = get_client()
    user_prompt = build_user_prompt(operator_notes)
    if feedback:
        user_prompt += (
            "\n\nYour previous output was rejected by the validator for this reason:\n"
            f"{feedback}\nReturn a corrected JSON array only."
        )

    response = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )

    text = response.output_text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM output was not valid JSON: {e}. Raw output: {text[:500]}")
