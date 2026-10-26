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

"""Semantic-recall fallback of ``search_history_text`` (Laya decision engine):

when keyword search finds nothing and a semantic backend is active, step
summaries that *mean* the same thing as the query are recalled.
"""

from types import SimpleNamespace

from artemis.tools.history.search import search_history_text


class FakeReader:
    def __init__(self, steps):
        self._steps = steps
        self.session_start_time = None
        self.base_dir = None

    def get_agent_friendly_steps(self):
        return self._steps

    def get_history_chunks(self):
        return []


class FakeSemanticEngine:
    semantic = True
    min_confidence = 0.72
    rescue_limit = 120

    def __init__(self, scores):
        self.scores = scores

    @property
    def backend_name(self):
        return "fake"

    def rank(self, query, texts):
        return [self.scores.get((query, text), 0.0) for text in texts]


def _steps():
    return [
        {
            "step_number": 1,
            "step_id": "s1",
            "summary": "Opened the WiFi settings page.",
        },
        {
            "step_number": 2,
            "step_id": "s2",
            "summary": "Enabled airplane mode from quick settings.",
        },
    ]


def test_semantic_recall_when_keywords_miss(monkeypatch):
    engine = FakeSemanticEngine({("connect to wireless network", "Opened the WiFi settings page."): 0.9})
    import artemis.decisions.engine as engine_module

    monkeypatch.setattr(engine_module, "get_decision_engine", lambda: engine)
    text = search_history_text(FakeReader(_steps()), query="connect to wireless network")
    assert "semantic recall via fake" in text
    assert "Opened the WiFi settings page." in text
    assert "Step 1" in text


def test_no_recall_without_semantic_backend(monkeypatch):
    engine = FakeSemanticEngine({})
    engine.semantic = False
    import artemis.decisions.engine as engine_module

    monkeypatch.setattr(engine_module, "get_decision_engine", lambda: engine)
    text = search_history_text(FakeReader(_steps()), query="connect to wireless network")
    assert "semantic recall" not in text
    assert "No matches" in text


def test_low_confidence_summaries_not_recalled(monkeypatch):
    engine = FakeSemanticEngine({("connect to wireless network", "Opened the WiFi settings page."): 0.3})
    import artemis.decisions.engine as engine_module

    monkeypatch.setattr(engine_module, "get_decision_engine", lambda: engine)
    text = search_history_text(FakeReader(_steps()), query="connect to wireless network")
    assert "semantic recall" not in text


def test_keyword_hits_short_circuit_recall(monkeypatch):
    engine = FakeSemanticEngine({("wifi", "Opened the WiFi settings page."): 0.9})
    import artemis.decisions.engine as engine_module

    monkeypatch.setattr(engine_module, "get_decision_engine", lambda: engine)
    text = search_history_text(FakeReader(_steps()), query="wifi")
    assert "semantic recall" not in text
    assert "WiFi" in text
