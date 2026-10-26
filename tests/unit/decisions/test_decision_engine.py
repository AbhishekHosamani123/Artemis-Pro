# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the Laya decision engine (backend selection, caching,
circuit breaking)."""

import pytest

from artemis.config import settings
from artemis.decisions.backends import DecisionBackendError
from artemis.decisions.engine import DecisionEngine
from artemis.decisions.lexical import lexical_similarity


class FakeSemanticBackend:
    name = "fake"
    semantic = True

    def __init__(self, scores=None, fail=False):
        self.scores = scores or {}
        self.fail = fail
        self.calls = 0

    def health(self):
        return True, "fake backend"

    def rank(self, query, texts):
        self.calls += 1
        if self.fail:
            raise DecisionBackendError("boom")
        return [self.scores.get((query, text), 0.0) for text in texts]

    def close(self):
        pass


@pytest.fixture
def auto_mode(monkeypatch):
    monkeypatch.setattr(settings, "ARTEMIS_DECISION_BACKEND", "auto")
    return monkeypatch


def test_rank_returns_backend_scores_and_caches():
    backend = FakeSemanticBackend({("power", "Battery"): 0.9})
    engine = DecisionEngine(backend=backend)

    first = engine.rank("power", ["Battery", "Volume"])
    assert first == [0.9, 0.0]
    assert engine.rank("power", ["Battery", "Volume"]) == first
    assert backend.calls == 1  # second identical call served from cache
    assert engine.semantic and engine.backend_name == "fake"


def test_backend_failure_circuit_breaks_to_zeros():
    backend = FakeSemanticBackend(fail=True)
    engine = DecisionEngine(backend=backend)

    assert engine.rank("power", ["Battery"]) == [0.0]
    assert not engine.semantic  # breaker trips: callers keep lexical results


def test_off_mode_disables_scoring(auto_mode):
    monkeypatch = auto_mode
    monkeypatch.setattr(settings, "ARTEMIS_DECISION_BACKEND", "off")
    engine = DecisionEngine(backend=FakeSemanticBackend({("q", "x"): 0.9}))
    assert engine.rank("q", ["x"]) == [0.0]
    assert engine.backend_name == "off"


def test_auto_mode_without_laya_falls_back_to_lexical(auto_mode, monkeypatch):
    monkeypatch.setattr(
        "artemis.decisions.engine.LayaBackend.health", lambda self: (False, "down")
    )
    engine = DecisionEngine()
    assert engine.backend_name == "lexical"
    assert not engine.semantic
    # Lexical floor: engine reports zeros (semantic channel), lexical scores
    # come from artemis.decisions.lexical directly.
    assert engine.rank("anything", ["nothing"]) == [0.0]


def test_rank_bounds_scores():
    backend = FakeSemanticBackend({("q", "x"): 5.0})
    assert DecisionEngine(backend=backend).rank("q", ["x"]) == [1.0]


def test_lexical_floor_is_untouched_by_engine():
    # Non-semantic engines never fake confidence: the caller's lexical
    # scores stay the single source of truth for the no-model path.
    assert lexical_similarity("settings", "Settings") == 1.0
