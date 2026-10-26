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

"""Laya decision engine: millisecond relevance scoring for element search
and history recall, with pluggable backends (laya-mlx server, OpenAI-
compatible selection call, or the always-available lexical scorer)."""

from artemis.decisions.engine import (
    DecisionBackendError,
    DecisionEngine,
    get_decision_engine,
    reset_decision_engine,
)
from artemis.decisions.lexical import lexical_similarity, normalize_text

__all__ = [
    "DecisionBackendError",
    "DecisionEngine",
    "get_decision_engine",
    "reset_decision_engine",
    "lexical_similarity",
    "normalize_text",
]
