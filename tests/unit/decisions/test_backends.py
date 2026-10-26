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

"""Wire-format tests for the decision-engine backends (offline, no network)."""

import httpx
import pytest

from artemis.decisions.backends import (
    DecisionBackendError,
    LayaBackend,
    OpenAICompatBackend,
    _extract_question_score,
)


def _mock_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_laya_response_parsing_top_level_float():
    assert _extract_question_score({"q0": 0.87, "usage": {}}, "q0") == pytest.approx(0.87)


def test_laya_response_parsing_nested_dict_shapes():
    assert _extract_question_score({"questions": {"q0": {"score": 0.4}}}, "q0") == pytest.approx(0.4)
    assert _extract_question_score({"answers": {"q0": {"p": True}}}, "q0") == 1.0
    assert _extract_question_score({"results": {"q0": {"value": "0.33"}}}, "q0") == pytest.approx(0.33)


def test_laya_response_parsing_rejects_garbage():
    with pytest.raises(DecisionBackendError):
        _extract_question_score({"questions": {"q0": {"unrelated": "x"}}}, "q0")


def test_laya_backend_speaks_systemone_wire_format():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = request.read()
        return httpx.Response(200, json={"questions": {"q0": {"score": 0.9}}})

    backend = LayaBackend(url="http://127.0.0.1:8081/v1/systemone", model="laya-421m")
    backend._client = _mock_client(handler)
    scores = backend.rank("power options", ["Battery", "Volume"])

    assert scores == [0.9, 0.9]
    assert captured["url"] == "http://127.0.0.1:8081/v1/systemone"
    body = __import__("json").loads(captured["body"])
    assert body["model"] == "laya-421m"
    assert body["state"] in ("Battery", "Volume")
    assert body["questions"]["q0"]["type"] == "noul"


def test_laya_server_error_raises_backend_error():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    backend = LayaBackend(url="http://127.0.0.1:8081")
    backend._client = _mock_client(handler)
    with pytest.raises(DecisionBackendError):
        backend.rank("q", ["x"])


def test_openai_selection_one_hot():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "2"}}]}
        )

    backend = OpenAICompatBackend(api_key="k", base_url="https://x/v1", model="m")
    backend._client = _mock_client(handler)
    assert backend.rank("power options", ["Battery", "Volume", "Airplane"]) == [0.0, 0.95, 0.0]


def test_openai_selection_zero_means_no_match():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "0 - none"}}]})

    backend = OpenAICompatBackend(api_key="k", base_url="https://x/v1", model="m")
    backend._client = _mock_client(handler)
    assert backend.rank("q", ["a", "b"]) == [0.0, 0.0]
