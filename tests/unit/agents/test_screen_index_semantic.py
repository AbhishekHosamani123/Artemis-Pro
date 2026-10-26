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

"""Semantic rescue pass of ``ScreenIndex.search_text`` (Laya decision engine)."""

import pytest

from artemis.agents.explorer.screen_index import ScreenIndex

W, H = 1080, 2400


def _node(text, bounds):
    return {"text": text, "bounds": bounds, "class": "android.widget.TextView"}


def _index():
    return ScreenIndex.from_hierarchy(
        [
            _node("Battery", "[100,100][300,200]"),
            _node("Volume: 60%", "[100,300][400,400]"),
            _node("Airplane mode", "[100,500][400,600]"),
        ],
        W,
        H,
    )


class FakeSemanticEngine:
    """Minimal stand-in for a semantic decision engine."""

    semantic = True
    min_confidence = 0.72
    rescue_limit = 120

    def __init__(self, scores):
        self.scores = scores

    def rank(self, query, texts):
        return [self.scores.get((query, text), 0.0) for text in texts]


def test_lexical_only_path_is_unchanged_without_engine():
    assert _index().search_text("power options", 0.7) == []
    assert [m.element.text for m in _index().search_text("Battery", 0.7)] == ["Battery"]


def test_semantic_rescue_surfaces_paraphrased_target():
    engine = FakeSemanticEngine({("power options", "Battery"): 0.92})
    matches = _index().search_text("power options", 0.7, decision_engine=engine)
    assert [m.element.text for m in matches] == ["Battery"]
    assert matches[0].score == pytest.approx(0.92)


def test_low_confidence_semantic_scores_are_ignored():
    engine = FakeSemanticEngine({("power options", "Battery"): 0.5})
    assert _index().search_text("power options", 0.7, decision_engine=engine) == []


def test_confident_lexical_match_skips_rescue():
    class CountingEngine(FakeSemanticEngine):
        def __init__(self):
            super().__init__({})
            self.calls = 0

        def rank(self, query, texts):
            self.calls += 1
            return [0.0] * len(texts)

    engine = CountingEngine()
    index = _index()
    matches = index.search_text("Battery", 0.7, decision_engine=engine)
    assert matches[0].element.text == "Battery"
    assert engine.calls == 0  # exact lexical hit: no model round trip


def test_rank_failure_keeps_lexical_results():
    class BrokenEngine(FakeSemanticEngine):
        def __init__(self):
            super().__init__({})

        def rank(self, query, texts):
            raise RuntimeError("backend exploded")

    index = _index()
    matches = index.search_text("Airplane", 0.7, decision_engine=BrokenEngine())
    assert [m.element.text for m in matches] == ["Airplane mode"]
