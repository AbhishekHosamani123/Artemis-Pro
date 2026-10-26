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

"""Unit tests for the decision engine's lexical scorer."""

import pytest

from artemis.decisions.lexical import lexical_similarity, normalize_text


@pytest.mark.parametrize(
    ("query", "text", "floor"),
    [
        ("gmail", "Gmail", 1.0),
        ("send", "Send message", 0.85),
        ("search", "Search", 1.0),
        ("battery saver mode", "Battery saver", 0.85),
    ],
)
def test_exact_and_substring_scores(query, text, floor):
    assert lexical_similarity(query, text) >= floor


@pytest.mark.parametrize(
    ("query", "text"),
    [
        ("ok", "Look here"),
        ("zzzz", "Settings"),
        ("", "Settings"),
        ("settings", ""),
    ],
)
def test_noise_does_not_match(query, text):
    assert lexical_similarity(query, text) < 0.85


def test_word_order_no_longer_defeats_matching():
    # "save battery" vs "battery saver": difflib alone scores ~0.5; token
    # overlap must lift reordered labels above the strict-search threshold.
    assert lexical_similarity("save battery", "battery saver") >= 0.7


def test_filler_words_and_punctuation_are_tolerated():
    assert lexical_similarity("battery settings", "Battery settings (advanced)") >= 0.85
    assert lexical_similarity("wifi", "Wi-Fi") >= 0.5
    assert lexical_similarity("battery 85", "Battery: 85%") >= 0.85


def test_normalization_still_available():
    assert normalize_text("  Send\n Message ") == "send message"


def test_similarity_range_is_bounded():
    for query, text in [("battery", "Battery 85%"), ("a", "a"), ("add", "account")]:
        assert 0.0 <= lexical_similarity(query, text) <= 1.0
