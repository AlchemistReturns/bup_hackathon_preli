# GridWise: 48-variable energy optimizer

An LLM interprets operator notes, deterministic guardrails validate every directive,
and an exact continuous linear program minimizes 24-hour grid cost. A separate
replay verifies the serialized response before the API returns it.

## Local setup (Python 3.12)

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
# Edit .env locally to supply OPENAI_API_KEY.
.venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

For production, install requirements.txt instead of requirements-dev.txt.

Required environment variable: OPENAI_API_KEY. Optional: OPENAI_MODEL
(default gpt-4o-mini, must support the Responses API, strict structured outputs
and temperature=0); LLM_MAX_RETRIES (0-2, default 2). All notes for one scenario
are interpreted together. Battery capacity accompanies the notes so percentage
reserves can be converted into kWh.

## Endpoints and sample checks

GET /health returns HTTP 200 with {"status":"ok"} when the application can serve
requests. This is local readiness; it does not make a paid model call or certify
provider quota/availability.

POST /optimize-energy accepts the published scenario schema and returns the
published interpretation, hourly plan, totals and summary. Invalid request JSON
or schema returns 400. Unavailable interpretation, infeasibility, solver failure,
replay failure or request timeout returns a controlled 500 without a schedule.

```sh
curl http://localhost:8000/health
curl -X POST http://localhost:8000/optimize-energy -H "Content-Type: application/json" --data-binary @sample_request.json
```

To obtain sample_request.json, save any case.input object from
tests/fixtures/public_cases.json. Or use the sample checker directly:

```powershell
# No credentials or network calls: solve using the public reference directives.
.\.venv\Scripts\python.exe scripts/check_samples.py
# Full API path, including real interpretation; requires a running server/key.
.\.venv\Scripts\python.exe scripts/check_samples.py --base-url http://localhost:8000
# Repeat the live run and save a complete per-case report (repeats can hit cache).
.\.venv\Scripts\python.exe scripts/check_samples.py --base-url http://localhost:8000 --repeat 2 --report live_results.json
# Offline optimizer timing (50 solves, including the first solve).
.\.venv\Scripts\python.exe scripts/check_samples.py --repeat 5
# Regression suite: live SDK transport is mocked; no paid calls.
.\.venv\Scripts\python.exe -m pytest -q
```

Use .venv/bin/python for these commands on Linux/macOS. The checker requires
each case to replay successfully and its cost to match the published optimum
within 0.01 BDT. Live checks separately compare the machine-readable interpretation,
replay against organizer ground truth, verify response consistency and compare cost.
The runner continues after failed cases and exits with a nonzero status if any fail.
Explanation wording and the particular optimal action sequence may differ.

## Optimizer

app/optimizer.py uses SciPy's in-process HiGHS LP solver. It has exactly 48
continuous variables before presolve: grid[h] and energy_after[h] for 24 hours.

With previous energy equal to initial_energy_kwh at hour zero:

```text
delta[h] = energy_after[h] - previous_energy[h]
solar_used[h] = demand[h] + delta[h] - grid[h]

minimize sum(tariff[h] * grid[h])

demand[h] - effective_solar[h] <= grid[h] - delta[h] <= demand[h]
-discharge_limit[h] <= delta[h] <= charge_limit[h]
active_reserve[h] <= energy_after[h] <= capacity
0 <= grid[h] <= active_grid_cap[h]  (no upper bound if uncapped)
energy_after[23] = initial_energy_kwh
```

The sparse coefficient matrices are constructed once at import. Bounds, costs
and right-hand sides are request-local, making concurrent requests independent.
The solver must report an optimal finite solution. Battery actions are derived
from signed energy differences, so no simultaneous charge/discharge variables or
binary switches exist. Free solar can be curtailed; grid export is prohibited.
The lossless battery model follows the challenge; efficiency losses are not an
omitted feature. Peak grid use is reported, not added to the cost objective.

The optimizer's directive compiler and app/replay.py are independent.
Replay recomputes physics and checks directives directly. It also checks the
JSON-round-tripped response and its six-decimal reported totals.

## Interpretation and reliability

The Responses API requests a strict JSON schema, then guardrails independently
check mappings, supported types, finite numeric ranges, sorted unique hours,
exact adjustment fields and applies/no_op semantics.

The model returns a compact internal schema: d is the directive list, t the type,
w the clock windows (s=start, e=end), and v the numeric adjustment or null.
app/llm_protocol.py expands this into typed directives and generates explanations
locally, reducing generated tokens without changing the public response schema.
The model extracts start_hour and end_hour clock boundaries through that protocol.
app/time_windows.py deterministically expands each window using start-inclusive,
end-exclusive intervals. The public response still contains hours; time_windows
is internal only. For example, 9 AM until 11 AM becomes [9, 10]. This prevents
the model's hour-list generation from accidentally including the stopping hour.
The model remains responsible for understanding the note and its clock times;
there are no sample-ID or phrase lookup shortcuts.

No provider error is converted to no_op. When the bounded interpretation retry
budget is exhausted, optimization does not run. Provider exception text, raw
model output and stack traces are not returned or logged by application handlers.
SDK retries are disabled; LangGraph alone owns up to two retries.

Each model attempt has a 6-second HTTP timeout, the interpretation stage has a
20-second budget, HiGHS has a 5-second solve limit, and the API has a 27-second
response deadline. Blocking work runs in the thread pool to keep /health
responsive. A timed-out worker may finish in the background; its result is
discarded, and model/solver limits still apply. Deadline settings are constants
in app/config.py. Successful requests log stage timings without notes or keys.

The OpenAI SDK was updated because the original 1.51.0 pin predates the Responses
API. The existing model selection is preserved.

## Latency and interpretation caching

The model client reuses HTTP connections. All notes share one model call, with a
compact strict output schema. The 48-variable solver and independent replay still
run on every request; no schedule or cost is cached.

Only successful, guardrail-validated interpretations are cached in memory. The key
includes exact note text and order, battery capacity, model and protocol version.
Identical concurrent requests share one in-flight interpretation. Returned objects
are isolated from mutation. Errors are never cached, and unrelated keys do not
wait on each other's network calls. The cache is per server worker, not shared
across machines, and is cleared by restart.

Optional settings: LLM_CACHE_SIZE (default 256 entries; 0 disables caching) and
LLM_CACHE_TTL_SECONDS (default 900 seconds). Set LLM_CACHE_SIZE=0 before starting
the server to measure fresh LLM latency; repeating client requests alone does not
bypass the server cache. A cache miss uses the real model, never a sample lookup.
Disable caching if your evaluation requires a fresh model call for every request.

Responses include X-Interpretation-Cache (miss/hit/shared/disabled) and Server-Timing
headers for interpretation, optimization and validation. The sample checker saves
these with per-request latency in its JSON report. Compare cache hits separately
from fresh-note requests; network/provider load can still cause latency spikes.

## Docker

```sh
docker build -t gridwise:48-variable .
docker run --rm -p 8000:8000 --env-file .env gridwise:48-variable
curl http://localhost:8000/health
```

The image binds to 0.0.0.0:8000; .env, virtual environments and Git metadata are
excluded. Docker Compose remains supported. Before submission, push the built
image to your registry, record its exact tag/digest and verify docker pull/run
from another machine. A registry push and public deployment are not performed
by this source update.

## Tests and known limits

The suite covers the ten public cases and optimal costs, a separate 120-variable
reference LP on 60 seeded randomized scenarios, surplus solar, flat/zero tariffs,
zero battery rates, terminal infeasibility, concurrent solves, corrupted plans,
non-finite inputs, percentage context, structured SDK requests, retry exhaustion,
HTTP validation and response deadlines.

Offline tests use organizer ground-truth directives and mocked model responses;
they do not establish real-model paraphrase accuracy, live provider latency,
public reachability or Docker runtime compatibility. Run the live sample checker
and paraphrase tests before submission. Boundary regression tests cover midnight,
single-hour and disjoint periods, malformed clock values, and the observed
extra-ending-hour errors.

Overlapping reserve and grid caps use the strictest bound. The existing
multiplicative interpretation for overlapping solar reductions is preserved.
The supplied specification does not clearly define that overlap: confirm it
with organizers if such cases are in scope.

## Dependencies and credits

FastAPI/Starlette: HTTP API and request execution.
Pydantic: request/response validation.
NumPy/SciPy/HiGHS: numerical LP solution.
OpenAI Python SDK: model access.
LangGraph: bounded interpretation/validation/retry control.
python-dotenv: local configuration.
pytest/httpx: tests and mocked SDK transport.
The public fixture is the organizer-provided BUP CSE Fest 2026 sample pack.

Reference documentation:
- [SciPy HiGHS interface](https://docs.scipy.org/doc/scipy-1.14.1/reference/optimize.linprog-highs.html)
- [OpenAI Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs)
