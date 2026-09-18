from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import Mock
import copy

import pytest
from fastapi.testclient import TestClient

from app import graph, main
from app.guardrails import GuardrailError, validate_all
from app.interpretation_cache import InterpretationCache
from app.llm_protocol import expand_compact_directives
from test_optimizer import CASES


def directives():
    return validate_all(expand_compact_directives({"d": [
        {"t": "no_charge", "w": [{"s": 9, "e": 11}], "v": None}
    ]}), 1, 200)


def test_cache_isolation_expiry_and_lru():
    clock = Mock(return_value=0)
    cache = InterpretationCache(maxsize=2, ttl=10, clock=clock)
    compute = Mock(side_effect=directives)
    first, source = cache.get_or_compute("a", compute, 1)
    assert source == "miss"
    first[0].structured_adjustment["hours"].append(23)
    second, source = cache.get_or_compute("a", compute, 1)
    assert source == "hit"
    assert second[0].structured_adjustment["hours"] == [9, 10]
    cache.get_or_compute("b", compute, 1)
    cache.get_or_compute("a", compute, 1)  # touch a, making b oldest
    cache.get_or_compute("c", compute, 1)
    assert cache.get_or_compute("b", compute, 1)[1] == "miss"
    clock.return_value = 11
    assert cache.get_or_compute("b", compute, 1)[1] == "miss"
    assert compute.call_count == 5


def test_cache_failure_not_saved_and_disabled():
    cache = InterpretationCache()
    compute = Mock(side_effect=[ValueError("failed"), directives()])
    with pytest.raises(ValueError):
        cache.get_or_compute("a", compute, 1)
    assert cache.get_or_compute("a", compute, 1)[1] == "miss"
    cache = InterpretationCache(maxsize=0)
    compute = Mock(side_effect=directives)
    for _ in range(2):
        assert cache.get_or_compute("a", compute, 1)[1] == "disabled"
    assert compute.call_count == 2


def test_singleflight_and_waiter_timeout():
    cache = InterpretationCache()
    started, finish = Event(), Event()
    def compute():
        started.set()
        assert finish.wait(3)
        return directives()
    call = Mock(side_effect=compute)
    with ThreadPoolExecutor(2) as pool:
        owner = pool.submit(cache.get_or_compute, "a", call, 2)
        assert started.wait(2)
        # A duplicate waits on the existing future, never issuing a second call.
        with pytest.raises(TimeoutError):
            cache.get_or_compute("a", call, 0.01)
        future = cache._pending["a"]
        original = future.result
        waiting = Event()
        def result(timeout=None):
            waiting.set()
            return original(timeout)
        future.result = result
        follower = pool.submit(cache.get_or_compute, "a", call, 2)
        assert waiting.wait(2)
        finish.set()
        assert owner.result()[1] == "miss"
        assert follower.result()[1] == "shared"
    assert call.call_count == 1


def test_cache_key_includes_capacity_note_order_and_model(monkeypatch):
    call = Mock(side_effect=lambda *a: directives())
    monkeypatch.setattr(graph, "_run_uncached", call)
    for notes, capacity in [(["a", "b"], 200), (["a", "b"], 200),
                            (["b", "a"], 200), (["a", "b"], 300)]:
        graph.run_interpretation(notes, capacity)
    assert call.call_count == 3
    monkeypatch.setattr(graph, "OPENAI_MODEL", "different-model")
    graph.run_interpretation(["a", "b"], 200)
    assert call.call_count == 4


@pytest.mark.parametrize("kind,value", [
    ("solar", 0.5), ("reserve", 100), ("no_charge", None),
    ("no_discharge", None), ("grid_cap", 25),
])
def test_compact_types_preserve_exclusive_end(kind, value):
    result = validate_all(expand_compact_directives({"d": [
        {"t": kind, "w": [{"s": 18, "e": 21}], "v": value}
    ]}), 1, 200)
    assert result[0].structured_adjustment["hours"] == [18, 19, 20]
    assert result[0].explanation


@pytest.mark.parametrize("item", [
    {"t": "invalid", "w": [], "v": None},
    {"t": "ignore", "w": [{"s": 0, "e": 1}], "v": None},
    {"t": "no_charge", "w": [], "v": 1},
    {"t": "reserve", "w": [], "v": True},
    {"t": "reserve", "w": [], "v": "10"},
    {"t": "solar", "w": [{"s": 0, "e": 1}], "v": float("nan")},
    {"t": "grid_cap", "w": [{"s": 0, "e": 1}], "v": -1},
    {"t": "no_charge", "w": [{"s": 0, "e": 25}], "v": None},
])
def test_bad_compact_output_rejected(item):
    with pytest.raises(GuardrailError):
        validate_all(expand_compact_directives({"d": [item]}), 1, 200)


def test_cache_hit_still_solves_and_replays_current_request(monkeypatch):
    case = copy.deepcopy(CASES[0])
    call = Mock(return_value=case["expected_output"]["directive_interpretation"])
    monkeypatch.setattr(graph, "interpret_notes", call)
    solve, replay = Mock(wraps=main.solve), Mock(wraps=main.replay_response)
    monkeypatch.setattr(main, "solve", solve)
    monkeypatch.setattr(main, "replay_response", replay)
    with TestClient(main.app) as client:
        first = client.post("/optimize-energy", json=case["input"])
        assert first.status_code == 200
        assert first.headers["x-interpretation-cache"] == "miss"
        # Different tariffs must not reuse the earlier schedule/cost.
        for hour in case["input"]["hours"]:
            hour["tariff_bdt_per_kwh"] *= 2
        second = client.post("/optimize-energy", json=case["input"])
        assert second.status_code == 200
        assert second.headers["x-interpretation-cache"] == "hit"
        assert "optimizer;dur=" in second.headers["server-timing"]
        assert second.json()["total_cost_bdt"] == pytest.approx(first.json()["total_cost_bdt"] * 2)
    assert call.call_count == 1
    assert solve.call_count == 2
    assert replay.call_count >= 2
