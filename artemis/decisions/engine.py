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

"""The Laya decision engine: fast typed decisions for Artemis hot paths.

Artemis repeatedly needs small relevance verdicts — *which UI label matches
the requested control?*, *which past step mentions X?* — that today cost a
full LLM/VLM round trip whenever lexical matching is unsure. This engine
answers them with a tiny decision model ("Laya-style": a single forward
pass per yes/no question) or falls back to an upgraded local scorer.

Backend selection (``ARTEMIS_DECISION_BACKEND``):

- ``auto`` (default) — use a running `laya-mlx <https://github.com/mizorewww/laya-mlx>`_
  server when one responds at ``ARTEMIS_LAYA_URL`` (free, milliseconds),
  otherwise the lexical scorer. Never spends API tokens.
- ``laya`` — require the Laya server; lexical if it is down.
- ``openai`` — explicit opt-in to one small selection call per rescue on an
  OpenAI-compatible endpoint (``OPENAI_API_KEY`` / ``OPENAI_BASE_URL``,
  model ``ARTEMIS_DECISION_OPENAI_MODEL``).
- ``lexical`` / ``off`` — never semantic; ``off`` disables scoring entirely.

The engine never raises: if the active backend fails, it circuit-breaks to
the lexical floor for a cooldown window and reports zeros (callers then
keep whatever lexical matching already produced).
"""

from __future__ import annotations

import threading
import time

from artemis.config import settings
from artemis.decisions.backends import (
    DecisionBackendError,
    LexicalBackend,
    LayaBackend,
    OpenAICompatBackend,
)
from artemis.decisions.lexical import lexical_similarity
from third_party.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)

#: When a semantic backend fails, skip it for this long before re-probing.
_BREAKER_COOLDOWN_S = 300.0

#: How often ``auto``/``laya`` re-probe a Laya server that was down, so a
#: server started later is picked up without a restart.
_LAYA_REPROBE_S = 60.0

#: Pair-cache cap for per-item semantic scores.
_CACHE_MAX_ENTRIES = 8192


class DecisionEngine:
    """Scores ``query``/``text`` relevance via the best configured backend.

    ``rank`` returns one relevance score in ``[0, 1]`` per input text. For
    non-semantic backends those scores are lexical; for semantic backends
    they are model probabilities. Callers merge semantic scores with their
    own lexical scores (``max``) so exact matches always win.
    """

    def __init__(self, backend=None):
        self._lock = threading.Lock()
        self._backend = backend
        self._resolved_at = time.monotonic()
        self._breaker_until = 0.0
        self._pair_cache: dict[tuple[str, str], float] = {}
        self._set_cache: dict[tuple[str, tuple[str, ...]], list[float]] = {}

    # ------------------------------------------------------------------ #
    # Backend resolution
    # ------------------------------------------------------------------ #

    @property
    def mode(self) -> str:
        return (getattr(settings, "ARTEMIS_DECISION_BACKEND", "") or "auto").strip().lower()

    def _backend_or_none(self) -> object | None:
        if self.mode in ("off",):
            return None
        with self._lock:
            now = time.monotonic()
            if self._backend is not None and not self._needs_resolve(now):
                return self._backend
            if self._backend is not None and not isinstance(self._backend, LexicalBackend):
                self._backend.close()
            self._backend, self._resolved_at = self._resolve(now)
            return self._backend

    def _needs_resolve(self, now: float) -> bool:
        if self.mode not in ("auto", "laya"):
            return False
        if isinstance(self._backend, LexicalBackend) and now - self._resolved_at >= _LAYA_REPROBE_S:
            # A Laya server may have been started since the last probe.
            return True
        return now - self._resolved_at >= _BREAKER_COOLDOWN_S

    def _resolve(self, now: float):
        mode = self.mode
        if mode == "openai":
            api_key = None
            try:
                secret = settings.get_api_key("openai")
                api_key = secret.get_secret_value() if secret else None
            except Exception:  # pylint: disable=broad-exception-caught
                api_key = None
            if api_key:
                backend = OpenAICompatBackend(
                    api_key=api_key,
                    base_url=getattr(settings, "OPENAI_BASE_URL", None),
                    model=getattr(settings, "ARTEMIS_DECISION_OPENAI_MODEL", "") or "gemini-3.6-flash",
                    timeout=float(getattr(settings, "ARTEMIS_DECISION_TIMEOUT", 3.0) or 3.0) + 3.0,
                )
                return backend, now
            logger.warning(
                "ARTEMIS_DECISION_BACKEND=openai but no OpenAI-compatible key; "
                "falling back to lexical scoring"
            )
            return LexicalBackend(), now

        if mode in ("auto", "laya"):
            laya = LayaBackend(
                url=getattr(settings, "ARTEMIS_LAYA_URL", "") or "http://127.0.0.1:8081",
                model=getattr(settings, "ARTEMIS_DECISION_MODEL", "") or "laya-421m",
                api_key=_secret_value(getattr(settings, "ARTEMIS_LAYA_API_KEY", None)),
                timeout=float(getattr(settings, "ARTEMIS_DECISION_TIMEOUT", 3.0) or 3.0),
            )
            healthy, detail = laya.health()
            if healthy or mode == "laya":
                if not healthy:
                    logger.warning(f"ARTEMIS_DECISION_BACKEND=laya but {detail}; using lexical")
                    laya.close()
                    return LexicalBackend(), now
                return laya, now
            laya.close()
            return LexicalBackend(), now

        return LexicalBackend(), now

    # ------------------------------------------------------------------ #
    # Public surface
    # ------------------------------------------------------------------ #

    @property
    def backend_name(self) -> str:
        backend = self._backend_or_none()
        return getattr(backend, "name", "off") if backend else "off"

    @property
    def semantic(self) -> bool:
        """True when a model-backed backend is active and healthy."""
        backend = self._backend_or_none()
        if not backend:
            return False
        if time.monotonic() < self._breaker_until:
            return False
        return bool(getattr(backend, "semantic", False))

    @property
    def min_confidence(self) -> float:
        value = float(getattr(settings, "ARTEMIS_DECISION_MIN_CONFIDENCE", 0.72) or 0.72)
        return max(0.0, min(1.0, value))

    @property
    def rescue_limit(self) -> int:
        return max(1, int(getattr(settings, "ARTEMIS_DECISION_RESCUE_LIMIT", 120) or 120))

    def rank(self, query: str, texts: list[str]) -> list[float]:
        """Relevance score per text; zeros when no semantic backend applies.

        Never raises and never blocks longer than the configured timeout:
        any backend failure circuit-breaks the engine and yields zeros, so
        callers simply keep their lexical results.
        """
        texts = list(texts)
        if not texts or self.mode == "off":
            return [0.0] * len(texts)
        try:
            return self._rank_cached(query, texts)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            self._trip_breaker(exc)
            return [0.0] * len(texts)

    def _rank_cached(self, query: str, texts: list[str]) -> list[float]:
        backend = self._backend_or_none()
        if backend is None or not getattr(backend, "semantic", False):
            return [0.0] * len(texts)

        key = (query, tuple(texts))
        cached = self._set_cache.get(key)
        if cached is not None and len(cached) == len(texts):
            return list(cached)

        if isinstance(backend, OpenAICompatBackend):
            scores = backend.rank(query, texts)
        else:
            scores = self._pair_scores(backend, query, texts)

        if len(self._set_cache) >= _CACHE_MAX_ENTRIES:
            self._set_cache.clear()
        self._set_cache[key] = list(scores)
        return scores

    def _pair_scores(self, backend, query: str, texts: list[str]) -> list[float]:
        missing = [t for t in texts if (query, t) not in self._pair_cache]
        if missing:
            for text, score in zip(missing, backend.rank(query, missing)):
                if len(self._pair_cache) >= _CACHE_MAX_ENTRIES:
                    self._pair_cache.clear()
                self._pair_cache[(query, text)] = _sanitize_score(score)
        return [self._pair_cache[(query, t)] for t in texts]

    def _trip_breaker(self, exc: Exception) -> None:
        self._breaker_until = time.monotonic() + _BREAKER_COOLDOWN_S
        logger.warning(
            f"Decision backend {self.backend_name} failed ({exc.__class__.__name__}: {exc}); "
            f"lexical scoring only for {int(_BREAKER_COOLDOWN_S)}s"
        )

    # ------------------------------------------------------------------ #
    # Introspection (CLI / doctor)
    # ------------------------------------------------------------------ #

    def info(self) -> dict:
        """Snapshot of the backend chain for status output and tests."""
        backend = self._backend_or_none()
        laya_health: tuple[bool, str] | None = None
        if isinstance(backend, LayaBackend):
            laya_health = backend.health()
        openai_key = bool(settings.get_api_key("openai"))
        return {
            "mode": self.mode,
            "backend": getattr(backend, "name", "off") if backend else "off",
            "semantic": self.semantic,
            "min_confidence": self.min_confidence,
            "rescue_limit": self.rescue_limit,
            "laya_url": getattr(settings, "ARTEMIS_LAYA_URL", ""),
            "laya": laya_health,
            "openai_configured": openai_key,
            "openai_base_url": getattr(settings, "OPENAI_BASE_URL", None) or "https://api.openai.com/v1",
            "openai_model": getattr(settings, "ARTEMIS_DECISION_OPENAI_MODEL", ""),
        }

    def close(self) -> None:
        with self._lock:
            if self._backend is not None and not isinstance(self._backend, LexicalBackend):
                self._backend.close()
            self._backend = None


def _secret_value(secret) -> str | None:
    if secret is None:
        return None
    try:
        value = secret.get_secret_value()  # pydantic SecretStr
    except AttributeError:
        value = str(secret)
    return value or None


def _sanitize_score(score) -> float:
    try:
        return max(0.0, min(1.0, float(score)))
    except (TypeError, ValueError):
        return 0.0


_ENGINE: DecisionEngine | None = None
_ENGINE_LOCK = threading.Lock()


def get_decision_engine() -> DecisionEngine:
    """Process-wide engine singleton (lazily created, thread-safe)."""
    global _ENGINE  # pylint: disable=global-statement
    if _ENGINE is None:
        with _ENGINE_LOCK:
            if _ENGINE is None:
                _ENGINE = DecisionEngine()
    return _ENGINE


def reset_decision_engine() -> None:
    """Drop the singleton (used by tests and by ``artemis laya`` after config changes)."""
    global _ENGINE  # pylint: disable=global-statement
    with _ENGINE_LOCK:
        if _ENGINE is not None:
            _ENGINE.close()
        _ENGINE = None


__all__ = [
    "DecisionEngine",
    "DecisionBackendError",
    "LexicalBackend",
    "LayaBackend",
    "OpenAICompatBackend",
    "get_decision_engine",
    "reset_decision_engine",
    "lexical_similarity",
]
