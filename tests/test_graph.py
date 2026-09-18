"""Retry loop (feedback actually reaches the model) and SAFE FAILURE (Section 08), driven through the REAL
interpret_notes -> graph -> guardrails code with a fake OpenAI client. No network, no key."""
import json

import pytest

import app.llm_client as llm_client
from app.config import LLM_MAX_RETRIES
from app.graph import node_fallback, run_interpretation
from app.guardrails import GuardrailError, validate_all

NOTES = ["Do not charge between 2 PM and 4 PM.", "The cafeteria menu changes tomorrow."]
GOOD = [
    {"note_index": 0, "applies": True, "directive_type": "no_charge_window",
     "structured_adjustment": {"hours": [14, 15]}, "explanation": "e"},
    {"note_index": 1, "applies": False, "directive_type": "no_op", "structured_adjustment": None, "explanation": "e"},
]
BAD_TYPE = [{**GOOD[0], "directive_type": "battery_boost"}, GOOD[1]]


class FakeClient:
    """Stands in for openai.OpenAI: .responses.create(...) returns scripted text or raises scripted errors."""

    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.responses = self

    def create(self, **kw):
        self.calls.append(kw)
        item = self.script[min(len(self.calls) - 1, len(self.script) - 1)]  # repeat the last item forever
        if isinstance(item, Exception):
            raise item
        text = item if isinstance(item, str) else json.dumps(item)
        return type("R", (), {"output_text": text})()

    def user_prompt(self, i):
        return self.calls[i]["input"][1]["content"]

    def system_prompt(self, i):
        return self.calls[i]["input"][0]["content"]


@pytest.fixture
def fake(monkeypatch):
    def install(script):
        c = FakeClient(script)
        monkeypatch.setattr(llm_client, "_client", c)
        return c
    return install


def assert_all_no_op(out, n=len(NOTES)):
    assert len(out) == n
    for i, d in enumerate(out):
        assert d.note_index == i
        assert d.directive_type == "no_op" and d.applies is False and d.structured_adjustment is None
        assert isinstance(d.explanation, str) and d.explanation


# ---------------------------------------------------------------- retry loop
def test_happy_path_single_call_no_feedback(fake):
    c = fake([GOOD])
    out = run_interpretation(NOTES, 200)
    assert len(c.calls) == 1
    assert [d.directive_type for d in out] == ["no_charge_window", "no_op"]
    assert "rejected" not in c.user_prompt(0)


def test_guardrail_error_text_is_fed_back_on_retry(fake):
    c = fake([BAD_TYPE, GOOD])
    with pytest.raises(GuardrailError) as exc:
        validate_all(BAD_TYPE, len(NOTES), 200)
    expected_msg = str(exc.value)

    out = run_interpretation(NOTES, 200)
    assert len(c.calls) == 2
    assert "rejected" not in c.user_prompt(0)  # first attempt carries no feedback
    retry = c.user_prompt(1)
    assert expected_msg in retry  # the exact validator message, not just "try again"
    assert "battery_boost" in retry
    assert NOTES[0] in retry and NOTES[1] in retry  # notes are re-sent
    assert c.system_prompt(1) == c.system_prompt(0)
    assert [d.directive_type for d in out] == ["no_charge_window", "no_op"]


def test_invalid_json_error_is_fed_back(fake):
    c = fake(["this is not json", GOOD])
    run_interpretation(NOTES, 200)
    assert len(c.calls) == 2
    assert "not valid JSON" in c.user_prompt(1)


def test_llm_exception_text_is_fed_back(fake):
    c = fake([RuntimeError("upstream 503"), GOOD])
    out = run_interpretation(NOTES, 200)
    assert "upstream 503" in c.user_prompt(1)
    assert out[0].directive_type == "no_charge_window"


def test_each_retry_gets_the_latest_error_not_a_stale_one(fake):
    other = [GOOD[0], {**GOOD[1], "applies": True}]  # different failure: no_op with applies=true
    c = fake([BAD_TYPE, other, GOOD])
    run_interpretation(NOTES, 200)
    assert len(c.calls) == 3
    assert "battery_boost" in c.user_prompt(1)
    assert "no_op requires applies=false" in c.user_prompt(2)
    assert "battery_boost" not in c.user_prompt(2)


def test_markdown_fenced_json_is_accepted(fake):
    fake(["```json\n" + json.dumps(GOOD) + "\n```"])
    assert run_interpretation(NOTES, 200)[0].directive_type == "no_charge_window"


@pytest.mark.xfail(strict=True, reason="BUG_NULL_NO_FEEDBACK: model output `null` parses to None, which node_validate "
                                       "treats as an LLM-call failure with no error text, so the retry has no feedback")
def test_json_null_output_gets_feedback(fake):
    c = fake(["null", GOOD])
    run_interpretation(NOTES, 200)
    assert "rejected" in c.user_prompt(1)


# ---------------------------------------------------------------- safe failure
ALWAYS_BAD = {
    "invalid-json": "definitely not json",
    "network-error": RuntimeError("connection reset"),
    "unsupported-type": BAD_TYPE,
    "json-null": "null",
    "dict-not-list": {"note_index": 0},
    "empty-list": [],
    "wrong-length": [GOOD[0]],
    "string-json": '"hello"',
    "prose": "Sure! Here are the directives: 1) no charge 2) nothing",
}


@pytest.mark.parametrize("name", ALWAYS_BAD)
def test_persistent_failure_ends_in_all_no_op_after_all_retries(fake, name):
    c = fake([ALWAYS_BAD[name]])
    out = run_interpretation(NOTES, 200)
    assert_all_no_op(out)
    assert len(c.calls) == 1 + LLM_MAX_RETRIES  # exactly the configured number of attempts, then stop


@pytest.mark.parametrize("name,payload", [
    ("unhashable-type", [{**GOOD[0], "directive_type": ["no_op"]}, GOOD[1]]),
    ("non-string-explanation", [{**GOOD[0], "explanation": 123}, GOOD[1]]),
])
@pytest.mark.xfail(strict=True, reason="BUG_UNHASHABLE_TYPE / BUG_EXPLANATION_TYPE: a non-GuardrailError escapes "
                                       "the graph instead of being retried and then safe-failed. main.py now "
                                       "contains it (see test_api_safe_failure), but graph.run_interpretation "
                                       "itself still raises")
def test_crash_class_outputs_are_retried_then_safe_failed(fake, name, payload):
    fake([payload])
    assert_all_no_op(run_interpretation(NOTES, 200))


def test_recovery_on_last_allowed_attempt(fake):
    c = fake([BAD_TYPE] * LLM_MAX_RETRIES + [GOOD])
    out = run_interpretation(NOTES, 200)
    assert len(c.calls) == 1 + LLM_MAX_RETRIES
    assert out[0].directive_type == "no_charge_window"


def test_fallback_never_invents_a_type_and_covers_every_note():
    for n in (1, 2, 3):
        state = {"operator_notes": ["x"] * n, "battery_capacity": 1.0, "raw_output": None, "error": "e",
                 "attempts": 9, "validated": None, "safe_failed": False}
        out = node_fallback(state)
        assert state["safe_failed"] is True
        assert_all_no_op(out["validated"], n)


def test_no_op_with_wrong_shape_from_model_is_never_passed_through(fake):
    """A model that says no_op but attaches an adjustment must not reach the optimizer as a real directive."""
    bad = [{**GOOD[1], "note_index": 0, "structured_adjustment": {"hours": [1]}}, {**GOOD[1], "note_index": 1}]
    fake([bad])
    assert_all_no_op(run_interpretation(NOTES, 200))
