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
- minimum_battery_reserve: {"hours": [...], "minimum_energy_kwh": number} // If given as a percentage of battery capacity (e.g. 50% of 200 kWh = 100 kWh), calculate the numeric kWh.
- no_charge_window: {"hours": [...]}
- no_discharge_window: {"hours": [...]}
- max_grid_window: {"hours": [...], "max_grid_kwh": number}
- no_op: structured_adjustment must be null (use for notes that do not change the 24-hour energy schedule)

Rules:
- CRITICAL WINDOW FORMULA:
  For any window 'from X PM until Y PM', the mathematical formula is:
  start_hour = X + 12
  end_hour = Y + 12
  num_hours = end_hour - start_hour = Y - X
  hours = list(range(start_hour, end_hour))
  Examples:
  - "from 6 PM until 10 PM": X=6, Y=10 -> start=18, end=22. num_hours=4 -> hours: [18, 19, 20, 21] (length MUST be 4!)
  - "from 6 PM until 9 PM": X=6, Y=9 -> start=18, end=21. num_hours=3 -> hours: [18, 19, 20] (length MUST be 3!)
  - "from 7 PM until 10 PM": X=7, Y=10 -> start=19, end=22. num_hours=3 -> hours: [19, 20, 21] (length MUST be 3!)
  - "from 7 PM until 9 PM": X=7, Y=9 -> start=19, end=21. num_hours=2 -> hours: [19, 20] (length MUST be 2!)
  - "from 1 PM to 3 PM": X=1, Y=3 -> start=13, end=15. num_hours=2 -> hours: [13, 14] (length MUST be 2!)
- hours must be unique integers 0-23 in ascending order.
- Do not invent demand, solar, tariff, or battery parameters.
- Do not invent directive types outside the six listed above.
- Return applies=true for every directive except no_op, which must have applies=false.
- Output ONLY a JSON array, one object per note, in the same order as the notes, with fields:
  note_index, applies, directive_type, structured_adjustment, explanation.
No prose, no markdown fences, JSON array only.
"""


def build_user_prompt(operator_notes: List[str], battery_capacity: Optional[float] = None) -> str:
    lines = [f"{i}: {note}" for i, note in enumerate(operator_notes)]
    prompt = "Operator notes:\n" + "\n".join(lines)
    if battery_capacity is not None:
        prompt += f"\n\nBattery capacity: {battery_capacity} kWh"
    return prompt


def interpret_notes(
    operator_notes: List[str],
    feedback: Optional[str] = None,
    battery_capacity: Optional[float] = None,
) -> list:
    """Calls the OpenAI Responses API and returns the parsed raw JSON list.
    Raises ValueError if the response is not valid JSON.
    """
    client = get_client()
    user_prompt = build_user_prompt(operator_notes, battery_capacity=battery_capacity)
    if feedback:
        user_prompt += (
            "\n\nYour previous output was rejected by the validator for this reason:\n"
            f"{feedback}\nReturn a corrected JSON array only."
        )

    response = client.responses.create(
        model=OPENAI_MODEL,
        instructions=SYSTEM_PROMPT,
        input=user_prompt,
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
