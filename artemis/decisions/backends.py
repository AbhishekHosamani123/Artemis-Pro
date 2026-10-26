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

"""Backends of the Laya decision engine.

The engine answers one question many times per step: *how well does this
text (a UI label, a history summary) match this description?* Full LLM
turns answer it in seconds; these backends answer it in milliseconds.

- :class:`LayaBackend` — client for a `laya-mlx <https://github.com/mizorewww/laya-mlx>`_
  "System One" server (``POST /v1/systemone``), the same wire format the
  jgrep ecosystem speaks with ``--api laya``. A 421M encoder scores each
  yes/no question in a single forward pass (~7–15 ms). Free, local, no API
  key. The reference server binds to ``127.0.0.1:8081`` on an Apple Silicon
  host; point ``ARTEMIS_LAYA_URL`` at any machine that runs one.
- :class:`OpenAICompatBackend` — one small selection call against any
  OpenAI-compatible chat endpoint (OpenAI, OpenRouter, Oxion-style proxies).
  Opt-in only, because it spends API tokens.
- :class:`LexicalBackend` — the local lexical scorer from
  :mod:`artemis.decisions.lexical`; never semantic, never fails.
"""

from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor

import httpx

from artemis.decisions.lexical import lexical_similarity


class DecisionBackendError(RuntimeError):
    """Raised by a backend when scoring fails; the engine circuit-breaks."""


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


class LexicalBackend:
    """The always-available local scorer; ``semantic`` is False by definition."""

    name = "lexical"
    semantic = False

    def health(self) -> tuple[bool, str]:
        return True, "always available (no model, no network)"

    def rank(self, query: str, texts: list[str]) -> list[float]:
        return [lexical_similarity(query, text) for text in texts]

    def close(self) -> None:
        pass


class LayaBackend:
    """HTTP client for a laya-mlx System One decision server."""

    name = "laya"
    semantic = True

    #: jgrep's adapter serves the decision endpoint under this path.
    _ENDPOINT_SUFFIX = "/v1/systemone"

    #: Ask the probability-style ("noul") question type, as jgrep does.
    _QUESTION_TYPE = "noul"

    def __init__(self, url: str, model: str = "laya-421m", api_key: str | None = None, timeout: float = 3.0):
        url = (url or "").strip().rstrip("/") or "http://127.0.0.1:8081"
        if url.endswith(self._ENDPOINT_SUFFIX):
            url = url[: -len(self._ENDPOINT_SUFFIX)]
        self.base_url = url
        self.model = model or "laya-421m"
        self.timeout = max(0.2, timeout)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(timeout=self.timeout, headers=headers)
        self._pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="laya-score")

    def health(self) -> tuple[bool, str]:
        """Probe ``GET /health``; returns (healthy, detail)."""
        try:
            response = self._client.get(f"{self.base_url}/health", timeout=min(self.timeout, 1.5))
            response.raise_for_status()
            body = response.json()
            detail = f"{body.get('model', self.model)} on {self.base_url}"
            return True, detail
        except Exception as exc:  # pylint: disable=broad-exception-caught
            return False, f"no Laya server at {self.base_url} ({exc.__class__.__name__})"

    def rank(self, query: str, texts: list[str]) -> list[float]:
        """Score every text against ``query`` (one request per text, parallel)."""
        futures = [self._pool.submit(self._score_one, query, text) for text in texts]
        return [future.result() for future in futures]

    def _score_one(self, query: str, text: str) -> float:
        payload = {
            "model": self.model,
            "state": text,
            "questions": {
                "q0": {
                    "type": self._QUESTION_TYPE,
                    "instructions": f'Does this text match the description "{query}"?',
                }
            },
        }
        try:
            response = self._client.post(f"{self.base_url}{self._ENDPOINT_SUFFIX}", json=payload)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise DecisionBackendError(f"Laya server request failed: {exc}") from exc
        return _extract_question_score(response.json(), "q0")

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
        self._client.close()


class OpenAICompatBackend:
    """One selection call against an OpenAI-compatible chat endpoint.

    Instead of scoring every candidate (one call each), a single chat turn
    asks for the index of the best-matching item — one round trip per
    rescue regardless of how many labels are on screen. Semantic but
    token-billed, hence opt-in via ``ARTEMIS_DECISION_BACKEND=openai``.
    """

    name = "openai"
    semantic = True

    #: Above this many items a selection prompt degrades; callers cap first.
    MAX_ITEMS = 40

    def __init__(self, api_key: str, base_url: str | None, model: str, timeout: float = 6.0):
        self.api_key = api_key
        self.base_url = (base_url or "https://api.openai.com/v1").strip().rstrip("/")
        self.model = model
        self.timeout = max(0.5, timeout)
        self._client = httpx.Client(
            timeout=self.timeout, headers={"Authorization": f"Bearer {api_key}"}
        )

    def health(self) -> tuple[bool, str]:
        if not self.api_key:
            return False, "no OpenAI-compatible API key configured"
        return True, f"{self.model} via {self.base_url}"

    def rank(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        items = texts[: self.MAX_ITEMS]
        listing = "\n".join(f"{i + 1}. {text}" for i, text in enumerate(items))
        try:
            response = self._client.post(
                f"{self.base_url}/chat/completions",
                json={
                    "model": self.model,
                    "temperature": 0,
                    "max_tokens": 8,
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a strict relevance judge for UI element labels."
                                " Reply with ONLY the single number (1-"
                                f"{len(items)}) of the item that best matches the"
                                " description, or 0 if none match. No other text."
                            ),
                        },
                        {
                            "role": "user",
                            "content": f'Description: "{query}"\nItems:\n{listing}',
                        },
                    ],
                },
            )
            response.raise_for_status()
            reply = response.json()["choices"][0]["message"]["content"]
        except Exception as exc:  # pylint: disable=broad-exception-caught
            raise DecisionBackendError(f"OpenAI-compatible selection failed: {exc}") from exc

        match = re.search(r"\d+", str(reply))
        best = int(match.group()) if match else 0
        if not 1 <= best <= len(items):
            return [0.0] * len(texts)
        scores = [0.0] * len(texts)
        scores[best - 1] = 0.95
        return scores

    def close(self) -> None:
        self._client.close()


def _extract_question_score(payload: dict, qid: str) -> float:
    """Pull the answer for ``qid`` out of a System One response.

    The reference adapter returns ``agent.predict``'s result dict; the answer
    for a question sits under its id (top level, or under ``questions`` /
    ``answers`` / ``results``), as a float, a bool, or a dict with a score
    field. Parsing stays tolerant so minor upstream shape drift does not
    break the engine.
    """
    container = payload if isinstance(payload, dict) else {}
    node = container.get(qid)
    if node is None:
        for group in ("questions", "answers", "results"):
            nested = container.get(group)
            if isinstance(nested, dict) and qid in nested:
                node = nested[qid]
                break
    if node is None and qid in str(container):
        node = container

    if isinstance(node, bool):
        return 1.0 if node else 0.0
    if isinstance(node, (int, float)):
        return _clamp01(node)
    if isinstance(node, dict):
        for key in ("score", "probability", "p", "value", "prob", "noul", "answer", "result"):
            value = node.get(key)
            if isinstance(value, bool):
                return 1.0 if value else 0.0
            if isinstance(value, (int, float)):
                return _clamp01(value)
            if isinstance(value, str):
                parsed = _parse_score_text(value)
                if parsed is not None:
                    return parsed
    if isinstance(node, str):
        parsed = _parse_score_text(node)
        if parsed is not None:
            return parsed
    raise DecisionBackendError(f"Could not parse a score for question {qid!r} from Laya response")


def _parse_score_text(text: str) -> float | None:
    lowered = text.strip().lower()
    if lowered in ("yes", "true", "match"):
        return 1.0
    if lowered in ("no", "false", "no match"):
        return 0.0
    match = re.search(r"\d*\.?\d+", lowered)
    if match:
        value = float(match.group())
        return _clamp01(value / 100.0 if value > 1.0 else value)
    return None
