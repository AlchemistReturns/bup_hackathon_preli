"""Bounded TTL/LRU cache with single-flight for model-derived interpretations."""
from collections import OrderedDict
from concurrent.futures import Future
from threading import Lock
from time import monotonic

from app.schemas import DirectiveInterpretation


class InterpretationCache:
    def __init__(self, maxsize=256, ttl=900.0, clock=monotonic):
        self.maxsize, self.ttl, self.clock = maxsize, ttl, clock
        self._entries = OrderedDict()
        self._pending = {}
        self._lock = Lock()

    @staticmethod
    def _restore(frozen):
        # Each caller gets its own models and nested adjustment dictionaries.
        return [DirectiveInterpretation.model_validate_json(item) for item in frozen]

    def get_or_compute(self, key, compute, timeout):
        if self.maxsize == 0:
            return compute(), "disabled"
        with self._lock:
            found = self._entries.get(key)
            if found is not None:
                expires, frozen = found
                if expires > self.clock():
                    self._entries.move_to_end(key)
                    return self._restore(frozen), "hit"
                del self._entries[key]
            future = self._pending.get(key)
            owner = future is None
            if owner:
                future = Future()
                self._pending[key] = future
        if not owner:
            return self._restore(future.result(timeout=timeout)), "shared"
        try:
            # compute() must return guardrail-validated interpretations.
            frozen = tuple(item.model_dump_json() for item in compute())
            with self._lock:
                self._entries[key] = (self.clock() + self.ttl, frozen)
                self._entries.move_to_end(key)
                while len(self._entries) > self.maxsize:
                    self._entries.popitem(last=False)
            future.set_result(frozen)
            return self._restore(frozen), "miss"
        except BaseException as exc:
            # Failed/partial interpretation never enters the cache.
            future.set_exception(exc)
            raise
        finally:
            with self._lock:
                self._pending.pop(key, None)
