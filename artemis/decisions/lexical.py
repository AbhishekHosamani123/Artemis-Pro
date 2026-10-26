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

"""Zero-dependency lexical scorer — the always-available backend of the
Laya decision engine (:mod:`artemis.decisions`).

Extends the classic character-level ``difflib`` element matching with token
overlap and character-trigram similarity so word order, filler words and
punctuation noise stop defeating label search. Pure Python, deterministic,
no model and no network: this is the floor every other backend must beat.
"""

from __future__ import annotations

import difflib
import re

#: Minimum query length for substring matches to count as strong evidence;
#: shorter fragments ("ok", "a") match far too much.
SUBSTRING_MIN_CHARS = 3

#: Score granted to a substring match whose query is long enough to be
#: meaningful (see ``SUBSTRING_MIN_CHARS``).
SUBSTRING_SCORE = 0.85

#: Trigram overlap is the noisiest signal, so it is discounted slightly.
_TRIGRAM_DISCOUNT = 0.95

#: Two tokens count as the same word when one is a prefix of the other and
#: the shorter one is at least this long ("save" ≈ "saver", "battery" ≈
#: "batteries"); below that, fragments match far too much.
_TOKEN_PREFIX_MIN = 4

#: Prefix-relaxed token matches are weaker evidence than exact word
#: equality, so a full-overlap score via prefixes is capped slightly.
_PREFIX_MATCH_DISCOUNT = 0.92


def normalize_text(value: object) -> str:
    """Case-folds and collapses whitespace so visually equal labels compare equal."""
    return re.sub(r"\s+", " ", str(value or "")).strip().casefold()


def _trigrams(text: str) -> set[str]:
    return {text[i : i + 3] for i in range(len(text) - 2)}


def _token_prefix_match(a: str, b: str) -> bool:
    """True when ``a`` and ``b`` are prefix variants of one word ("save"/"saver")."""
    shorter = min(len(a), len(b))
    return shorter >= _TOKEN_PREFIX_MIN and a[:shorter] == b[:shorter]


def lexical_similarity(query: str, text: str) -> float:
    """Lexical similarity in ``[0, 1]`` between a query and an element label.

    Composes three signals and returns the strongest:

    - character-level ``difflib`` ratio (the historical baseline), with a
      minimum score for substring matches when the query is at least three
      characters long;
    - token-set Jaccard overlap, which keeps reordered labels
      ("save battery" vs "battery saver") and extra words ("Battery saver
      mode" vs "Battery saver") scoring high;
    - character-trigram overlap, which absorbs punctuation and formatting
      noise ("Wi-Fi" vs "wifi", "85%" vs "85 %").
    """
    q, t = normalize_text(query), normalize_text(text)
    if not q or not t:
        return 0.0
    if q == t:
        return 1.0

    ratio = difflib.SequenceMatcher(None, q, t).ratio()
    if len(q) >= SUBSTRING_MIN_CHARS and (q in t or t in q):
        ratio = max(ratio, SUBSTRING_SCORE)

    q_tokens, t_tokens = set(q.split()), set(t.split())
    exact_overlap = len(q_tokens & t_tokens)
    prefix_overlap = sum(
        1
        for qt in q_tokens - t_tokens
        if any(_token_prefix_match(qt, tt) for tt in t_tokens - q_tokens)
    )
    # Dice coefficient over tokens: full overlap between equal token counts
    # reaches 1.0, and shared-but-reordered labels ("save battery" vs
    # "battery saver") stay far above the strict-search threshold.
    overlap = exact_overlap + prefix_overlap
    token_score = (2.0 * overlap / (len(q_tokens) + len(t_tokens))) if (len(q_tokens) + len(t_tokens)) else 0.0
    if prefix_overlap:
        token_score *= _PREFIX_MATCH_DISCOUNT
    if len(q_tokens) == 1:
        # A lone shared token is weaker evidence than a multi-token overlap.
        token_score *= 0.9

    q_grams, t_grams = _trigrams(q), _trigrams(t)
    trigram = len(q_grams & t_grams) / len(q_grams | t_grams) if q_grams and t_grams else 0.0

    return min(1.0, max(ratio, token_score, trigram * _TRIGRAM_DISCOUNT))
