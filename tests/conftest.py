import pytest

from app import graph
from app.interpretation_cache import InterpretationCache


@pytest.fixture(autouse=True)
def isolate_interpretation_cache(monkeypatch):
    # Prevent test fixtures/mocked model replies from affecting other tests.
    monkeypatch.setattr(graph, "_cache", InterpretationCache())
