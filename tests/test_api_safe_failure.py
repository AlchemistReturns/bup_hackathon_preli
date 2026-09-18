"""Section 08 SAFE FAILURE at the HTTP boundary: whatever the model emits, POST /optimize-energy answers 200 with
all-no_op interpretation and a valid plan - never a 500, never a made-up directive. Uses the REAL graph + guardrails
with a fake OpenAI client (the live model is not involved)."""
import copy
import json

import pytest
from fastapi.testclient import TestClient

import app.llm_client as llm_client
import app.main as main
from helpers import find_cases_file
from test_graph import ALWAYS_BAD, GOOD, FakeClient
from test_response_schema import check_response

REQ = copy.deepcopy(json.loads(find_cases_file().read_text(encoding="utf-8"))["cases"][0]["input"])
REQ["operator_notes"] = ["Do not charge between 2 PM and 4 PM.", "The cafeteria menu changes tomorrow."]
client = TestClient(main.app, raise_server_exceptions=False)

CRASH_CLASS = {
    "unhashable-directive-type": [{**GOOD[0], "directive_type": ["no_op"]}, GOOD[1]],
    "non-string-explanation": [{**GOOD[0], "explanation": 123}, GOOD[1]],
    "dict-directive-type": [{**GOOD[0], "directive_type": {"a": 1}}, GOOD[1]],
}


@pytest.mark.parametrize("name", list(ALWAYS_BAD) + list(CRASH_CLASS))
def test_llm_garbage_yields_200_all_no_op_and_valid_plan(monkeypatch, name):
    script = ALWAYS_BAD[name] if name in ALWAYS_BAD else CRASH_CLASS[name]
    monkeypatch.setattr(llm_client, "_client", FakeClient([script]))
    r = client.post("/optimize-energy", json=REQ)
    body = check_response(r, REQ)
    assert [d["directive_type"] for d in body["directive_interpretation"]] == ["no_op", "no_op"]
    assert all(d["applies"] is False and d["structured_adjustment"] is None for d in body["directive_interpretation"])


def test_interpretation_pipeline_crash_is_contained(monkeypatch):
    def explode(**kw):
        raise RuntimeError("langgraph exploded")
    monkeypatch.setattr(main, "run_interpretation", explode)
    body = check_response(client.post("/optimize-energy", json=REQ), REQ)
    assert all(d["directive_type"] == "no_op" for d in body["directive_interpretation"])


def test_interpretation_returning_wrong_length_is_contained(monkeypatch):
    monkeypatch.setattr(main, "run_interpretation", lambda **kw: [])
    body = check_response(client.post("/optimize-energy", json=REQ), REQ)
    assert len(body["directive_interpretation"]) == 2


def test_no_stack_trace_or_secret_in_500s(monkeypatch):
    monkeypatch.setattr(main, "solve", lambda *a, **k: (_ for _ in ()).throw(KeyError("OPENAI_API_KEY=sk-secret")))
    monkeypatch.setattr(main, "run_interpretation", lambda **kw: [])
    r = client.post("/optimize-energy", json=REQ)
    # a bug in the solver path is the only thing left that may 500; when it does, the body must be generic
    if r.status_code == 500:
        assert "sk-secret" not in r.text and "Traceback" not in r.text
    else:
        assert r.status_code == 200
