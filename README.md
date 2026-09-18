# GridWise — Scaffold

## Run
```
cp .env.example .env          # set OPENAI_API_KEY
uv venv .venv --python 3.12   # pydantic-core has no wheel for 3.14 yet
uv pip install -r requirements.txt --python .venv
.venv\Scripts\uvicorn app.main:app --reload --port 8000
```

## Run with Docker
```
cp .env.example .env   # set OPENAI_API_KEY
docker build -t gridwise .
docker run --rm -p 8000:8000 --env-file .env gridwise
```
API at `http://localhost:8000`. Image uses `uv` internally to build the venv, same deps as local run.

Or with Compose:
```
cp .env.example .env   # set OPENAI_API_KEY
docker compose up -d --build
docker compose logs -f
docker compose down
```

## Structure
- `app/schemas.py` — Pydantic models, exact request/response contract (Section 07/10).
- `app/llm_client.py` — OpenAI Responses API call, system prompt for directive extraction.
- `app/guardrails.py` — deterministic validation of raw LLM JSON (Section 08).
- `app/graph.py` — LangGraph pipeline: `call_llm -> validate -> (retry | fallback | done)`.
  Retries up to `LLM_MAX_RETRIES` (default 2) feeding the guardrail error back to the LLM.
  On exhausted retries, fails safe: marks all notes `no_op` rather than crashing/inventing.
- `app/optimizer.py` — PuLP LP model (Section 09), applies validated directives as constraints.
- `app/replay.py` — independent recompute/validate of the final plan (Section 11.3), mirrors
  what the hidden judge does; a bad plan here returns HTTP 500 instead of shipping garbage.
- `app/main.py` — `GET /health`, `POST /optimize-energy`.

## Known scaffold shortcuts (fix once base API is confirmed working end-to-end)
- No round-trip battery efficiency loss modeled — LP is pure energy balance.
- No binary charge/discharge exclusivity constraint (relies on cost-minimality; code post-processes
  the rare simultaneous case as a fallback, see `optimizer.py`).
- `total_grid_kwh`/`total_cost_bdt`/`peak_grid_kwh` are taken from `replay()`'s recompute, not the
  solver's raw output, so they're guaranteed self-consistent with `hourly_plan`.
- No persistence/caching of LLM calls; every request re-calls the API.
- CBC (PuLP's bundled solver) is used; swap for a commercial solver if scenario size grows.

## Tested locally
`app/main.py`'s `/health` and `/optimize-energy` were smoke-tested with `TestClient` and a stubbed
LLM step (no live OpenAI call, since this sandbox has no network access to it) — full request →
directive interpretation → LP solve → replay → response cycle returns 200 with a consistent plan.
Swap in a real `OPENAI_API_KEY` and the LangGraph path runs unchanged.
