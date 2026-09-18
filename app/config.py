import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))
LLM_TIMEOUT_SECONDS = 6.0
LLM_BUDGET_SECONDS = 20.0
SOLVER_TIMEOUT_SECONDS = 5.0
REQUEST_TIMEOUT_SECONDS = 27.0
LLM_CACHE_SIZE = int(os.getenv("LLM_CACHE_SIZE", "256"))
LLM_CACHE_TTL_SECONDS = float(os.getenv("LLM_CACHE_TTL_SECONDS", "900"))

if not 0 <= LLM_CACHE_SIZE <= 10000:
    raise ValueError("LLM_CACHE_SIZE must be between 0 and 10000")
if not 0 < LLM_CACHE_TTL_SECONDS <= 86400:
    raise ValueError("LLM_CACHE_TTL_SECONDS must be within (0, 86400]")

if not 0 <= LLM_MAX_RETRIES <= 2:
    raise ValueError("LLM_MAX_RETRIES must be between 0 and 2")

DIRECTIVE_TYPES = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
}
