# GridWise: 48-variable energy optimizer

An LLM interprets operator notes, deterministic guardrails validate every directive,
and an exact continuous linear program minimizes 24-hour grid cost. A separate
replay independently recomputes and verifies the response before the API returns it.

```
operator notes ─▶ LLM interpretation ─▶ guardrails ─▶ optimizer (LP) ─▶ replay ─▶ response
                   (untrusted)           (deterministic)  (HiGHS)       (independent check)
```

---

## 1. Environment & configuration

Copy `.env.example` to `.env` and fill in your own key — never commit `.env`.

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `OPENAI_API_KEY` | yes | — | your OpenAI key; the app fails at startup without it being set (empty string default, calls fail) |
| `OPENAI_MODEL` | no | `gpt-4o-mini` | must support the selected API mode, strict structured outputs, and `temperature=0` |
| `OPENAI_API_MODE` | no | `chat` | `chat` (Chat Completions) or `responses` (Responses API); both use the same key/model |
| `LLM_MAX_RETRIES` | no | `2` | 0–2; LangGraph-owned retries on guardrail failure, feeding the error back to the model |
| `LLM_CACHE_SIZE` | no | `0` | 0 disables the in-memory interpretation cache; set >0 (e.g. `256`) to opt in |
| `LLM_CACHE_TTL_SECONDS` | no | `900` | cache entry lifetime when caching is enabled; range (0, 86400] |

All values are read once at import time in `app/config.py` (via `python-dotenv`) and
validated with hard failures (`ValueError`) for out-of-range settings, so a bad
`.env` fails fast at startup rather than mid-request.

Fixed (non-configurable) budgets, also in `app/config.py`:

| Constant | Value | Applies to |
|---|---|---|
| `LLM_TIMEOUT_SECONDS` | 6.0s | per model HTTP attempt |
| `LLM_BUDGET_SECONDS` | 20.0s | whole interpretation stage (all attempts) |
| `SOLVER_TIMEOUT_SECONDS` | 5.0s | HiGHS LP solve |
| `REQUEST_TIMEOUT_SECONDS` | 27.0s | whole `/optimize-energy` request |

Blocking work runs in the thread pool so `/health` stays responsive under load.
A timed-out worker may finish in the background, but its result is discarded and
the deadline still applies to the response.

### Model provider

- Provider: **OpenAI** only, via the official Python SDK (`openai==1.109.1` — bumped
  from the original `1.51.0` pin, which predates the Responses API).
- Two request modes, selected by `OPENAI_API_MODE`:
  - `chat` (default) — Chat Completions with a strict JSON schema response format.
  - `responses` — the Responses API, same schema/semantics.
- SDK-level retries are disabled; **LangGraph alone** owns the bounded retry loop
  (`LLM_MAX_RETRIES`), so a retry always carries the guardrail's rejection reason
  back to the model rather than blindly repeating the same request.
- No provider error is ever silently converted to `no_op`. Exhausting the retry
  budget raises `InterpretationError` and optimization does not run — the API
  returns a 500, never a guessed schedule.
- Provider exception text, raw model output, and stack traces are never returned
  to the client or logged — only stage timings for successful requests.

---

## 2. Architecture: LLM → guardrails → optimizer → replay

### 2.1 Interpretation (untrusted input)

`app/llm_client.py` sends all operator notes for a scenario in **one** model call,
using a strict JSON schema, requesting a compact numeric protocol rather than the
public field names directly (fewer generated tokens):

```
{"d": [[type, value, start, end, ...], ...]}
```

- `type`: 0=ignore, 1=solar, 2=reserve, 3=no_charge, 4=no_discharge, 5=grid_cap
- `value`: the numeric adjustment (0/unused for ignore, no_charge, no_discharge)
- `start`/`end` pairs: one or more disjoint clock windows

`app/llm_protocol.py` expands this into the public directive schema
(`solar_reduction`, `minimum_battery_reserve`, `no_charge_window`,
`no_discharge_window`, `max_grid_window`, `no_op`) and generates explanation text
locally — the model never writes free-text explanations, only structure.
`app/time_windows.py` deterministically converts clock boundaries to start-inclusive,
end-exclusive hour lists (e.g. "9 AM until 11 AM" → `[9, 10]`), so the model's
clock-boundary output can't accidentally include the stopping hour.

The model is never trusted directly. `app/guardrails.py` independently validates:
mapping shape, supported types, finite numeric ranges, sorted/unique hours 0–23,
exact required adjustment fields per type, and `applies`/`no_op` semantics. On
failure, `app/graph.py` (a LangGraph `call_llm → validate → retry|fail` pipeline)
retries up to `LLM_MAX_RETRIES` times with the guardrail error fed back to the
model. If retries are exhausted, the pipeline raises rather than guessing — no
directive is ever invented, and the request fails safe with a 500.

### 2.2 Optimization (`app/optimizer.py`)

SciPy's in-process HiGHS LP solver. Exactly 48 continuous variables before
presolve: `grid[h]` and `energy_after[h]` for 24 hours (no split charge/discharge
variables, no binary switches — battery action is derived from the signed energy
delta each hour).

```text
delta[h] = energy_after[h] - previous_energy[h]
solar_used[h] = demand[h] + delta[h] - grid[h]

minimize sum(tariff[h] * grid[h])

demand[h] - effective_solar[h] <= grid[h] - delta[h] <= demand[h]
-discharge_limit[h] <= delta[h] <= charge_limit[h]
active_reserve[h] <= energy_after[h] <= capacity
0 <= grid[h] <= active_grid_cap[h]      (no upper bound if uncapped)
energy_after[23] = initial_energy_kwh
```

Validated directives compile into per-hour parameters (effective solar, active
reserve, no-charge/no-discharge hours, active grid cap) before the LP is built.
Sparse coefficient matrices are constructed once at import; bounds/costs/RHS are
request-local, so concurrent requests are independent. The solver must report an
optimal, finite solution or the request fails (`OptimizationError`). Free solar
can be curtailed; grid export is prohibited. The battery model is lossless (no
round-trip efficiency loss) — this follows the challenge spec, not an oversight.
Peak grid use is reported but not part of the cost objective.

### 2.3 Replay / independent verification (`app/replay.py`)

Completely independent of the optimizer's own directive compiler. Recomputes
physics and checks directive compliance directly from the JSON-round-tripped
response — the same thing a hidden judge would do. Also verifies the response's
six-decimal reported totals match its own recomputation. If replay disagrees with
what the optimizer produced, the API returns 500 rather than shipping a plan that
looks fine but isn't self-consistent.

---

## 3. Local setup (Python 3.12)

PowerShell:
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
Copy-Item .env.example .env
# Edit .env locally and set your own OPENAI_API_KEY. Never commit it.
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Linux/macOS:
```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
cp .env.example .env
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

For production, install `requirements.txt` instead of `requirements-dev.txt`
(the dev file adds `pytest`/`httpx` for the test suite only).

Using `uv` instead of raw `pip` (faster, and pins the venv's Python explicitly):
```sh
uv venv .venv --python 3.12
uv pip install -r requirements-dev.txt --python .venv
```

---

## 4. Endpoints

`GET /health` — HTTP 200 `{"status":"ok"}` when the app can serve requests. This
is local readiness only; it does not make a paid model call or certify provider
quota/availability.

`POST /optimize-energy` — accepts the published scenario schema, returns the
published interpretation, hourly plan, totals, and summary.
- Invalid request JSON/schema → 400.
- Unavailable interpretation, infeasibility, solver failure, replay failure, or
  request timeout → controlled 500, never a partial or guessed schedule.

```sh
curl http://localhost:8000/health
curl -X POST http://localhost:8000/optimize-energy \
  -H "Content-Type: application/json" --data-binary @sample_request.json
```

`sample_request.json` can be any `case.input` object from
`tests/fixtures/public_cases.json`.

---

## 5. Public-sample test procedure and expected result

The organizer-provided BUP CSE Fest 2026 public sample pack (10 cases) lives at
`tests/fixtures/public_cases.json`.

```powershell
# Offline: solve using the public reference directives, no LLM/network/key needed.
.\.venv\Scripts\python.exe scripts/check_samples.py

# Live: full pipeline including real LLM interpretation. Requires a running
# server (see §3) and a valid OPENAI_API_KEY.
.\.venv\Scripts\python.exe scripts/check_samples.py --base-url http://localhost:8000

# Repeat the live run and save a full per-case report (repeats can hit cache
# if LLM_CACHE_SIZE>0).
.\.venv\Scripts\python.exe scripts/check_samples.py --base-url http://localhost:8000 --repeat 2 --report live_results.json

# Regression suite: SDK transport is mocked, no paid calls.
.\.venv\Scripts\python.exe -m pytest -q
```
(Linux/macOS: use `.venv/bin/python`.)

**Expected result:** `10/10 passed` on both the offline and live paths — every
case must replay successfully and its cost must match the published optimum
within 0.01 BDT. The live path additionally reports, per case:
`interpretation=True`, `ground_truth_valid=True`, `optimal_cost=True`, and a
`cache=` status (`hit`/`miss`/`disabled`). The runner continues past failed
cases and exits non-zero if any fail. Explanation wording and the particular
optimal action sequence are not required to match the reference byte-for-byte —
only cost, directive semantics, and physical consistency.

Last verified live run on `master`: 10/10 passed, `cache=disabled` on every
case (default config), median latency ≈3.5s, p95 ≈5.5s (single-run sample —
see §6 for caching behavior and its effect on latency).

Observed offline optimizer timing: median ≈1.2ms, p95 ≈2ms per case (pure LP
solve, no network).

---

## 6. Latency and the interpretation cache

`LLM_CACHE_SIZE` defaults to `0` — **caching is off by default**, every request
makes a fresh model call, including repeated identical inputs. Confirm this via
the `X-Interpretation-Cache` response header (`disabled` when off). An existing
environment variable overrides the default, so pin it explicitly if you need a
guaranteed-fresh-call environment (e.g. for judging):

```powershell
$env:LLM_CACHE_SIZE = "0"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

If enabled (`LLM_CACHE_SIZE` > 0): only successful, guardrail-validated
interpretations are cached in memory. Cache key = exact note text + order +
battery capacity + API mode + model + protocol version. Identical concurrent
requests share one in-flight interpretation (single-flight). Returned objects
are isolated from mutation. Errors are never cached. The cache is per
server-worker, not shared across machines or processes, and clears on restart.

Every response carries `X-Interpretation-Cache` (`miss`/`hit`/`shared`/`disabled`)
and `Server-Timing` (interpretation/optimization/validation stage timings)
headers. `scripts/check_samples.py` records both per request in its JSON report.

To compare two running no-cache versions head-to-head:
```powershell
.\.venv\Scripts\python.exe scripts/benchmark_uncached.py --baseline-url http://127.0.0.1:8002 --candidate-url http://127.0.0.1:8003 --repeat 2 --report comparison.json
```

---

## 7. Docker

### Build and run locally
```sh
docker build -t gridwise:latest .
docker run --rm -p 8000:8000 --env-file .env gridwise:latest
curl http://localhost:8000/health
```

Or with Compose:
```sh
docker compose up -d --build
docker compose logs -f
docker compose down
```

### Pull a published image (fallback if you don't want to build locally)
```sh
docker pull abrar19/gridwise:latest
docker run --rm -p 8000:8000 --env-file .env abrar19/gridwise:latest
curl http://localhost:8000/health
```
> Always verify the exact tag/digest you pulled matches what was published,
> and re-run the public sample check (§5) against the running container
> before trusting it.

The image binds to `0.0.0.0:8000`. `.env`, virtual environments, and Git
metadata are excluded from the build context via `.dockerignore` — the image
never bakes in your key; it's supplied at `docker run` time via `--env-file`/`-e`.

---

## 8. Dependencies

| Package | Purpose |
|---|---|
| FastAPI / Starlette | HTTP API and request execution |
| Pydantic | request/response schema validation |
| NumPy / SciPy (HiGHS) | numerical LP solution |
| OpenAI Python SDK | model access (chat or responses mode) |
| LangGraph | bounded interpretation/validation/retry control |
| python-dotenv | local `.env` configuration loading |
| pytest / httpx (dev only) | tests, mocked SDK transport |

Reference docs:
- [SciPy HiGHS interface](https://docs.scipy.org/doc/scipy-1.14.1/reference/optimize.linprog-highs.html)
- [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
- [OpenAI latency optimization](https://developers.openai.com/api/docs/guides/latency-optimization)

The public fixture (`tests/fixtures/public_cases.json`) is the organizer-provided
BUP CSE Fest 2026 sample pack.

---

## 9. Limitations

- **No round-trip battery efficiency loss** — the LP is pure energy balance
  (matches the challenge spec, not an omission).
- **No hidden judge access** — the offline/live checks here only validate
  against the *public* sample pack; passing 10/10 does not guarantee behavior
  on hidden judge cases.
- **LLM interpretation is inherently non-deterministic** — even with
  `temperature=0` and a strict schema, wording paraphrases can occasionally
  shift the model's directive reading; guardrails catch structurally invalid
  output, not semantically wrong-but-valid output.
- **Overlapping directive semantics are underspecified upstream.** Overlapping
  reserve and grid caps use the strictest bound; overlapping solar reductions
  use a multiplicative interpretation. The organizer spec doesn't clearly
  define this — confirm with organizers if such cases are in scope.
- **Cache is per-process** — not shared across horizontally scaled workers or
  machines; restarting a worker clears it.
- **Offline tests use mocked model responses** and organizer ground-truth
  directives — they don't establish real-model paraphrase accuracy, live
  provider latency, public reachability, or Docker runtime compatibility on
  their own. Run the live sample checker (§5) and a Docker pull/run (§7)
  before submission.
- **Published image freshness isn't automatic.** `docker pull abrar19/gridwise:latest`
  (§7) only reflects whatever was last pushed; if source changes after a push,
  rebuild and re-push before relying on the pulled image, and re-run §5 against
  the freshly pulled container to confirm it's current.

---

## 10. Secret handling

- `OPENAI_API_KEY` lives only in your local `.env`, which is git-ignored
  (`.gitignore`) and docker-ignored (`.dockerignore`) — it is never committed
  and never baked into a built image.
- Copy `.env.example` → `.env` and fill in your own key; `.env.example`
  contains only placeholder values, safe to commit.
- Supply the key to a running container at `docker run`/`docker compose up`
  time via `--env-file .env` (or `-e OPENAI_API_KEY=...`), never via `COPY`,
  `ARG`, or hardcoding in the `Dockerfile`.
- Application handlers never return or log provider exception text, raw model
  output, or stack traces — only stage timings for successful requests. Don't
  add logging that captures request/response bodies in a way that could leak
  the key or note content to a shared log sink without review.
- If a key is ever exposed (committed, logged, pasted into a shared channel),
  rotate it in the OpenAI dashboard immediately — treat the old value as
  compromised, don't just delete it from history.
- Before pushing a Docker image publicly, double-check with `docker history`
  or by inspecting layers that no `.env` or key ever entered the build context
  (the current `.dockerignore` already excludes it, but re-verify after any
  Dockerfile change).
