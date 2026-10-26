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

"""Unit tests for the on-device AI cursor firing path."""

import json
import urllib.error

import pytest

import artemis.drivers.android.ai_cursor as ai_cursor
from artemis.config import settings


class FakeSession:
    def __init__(self, base_url="http://127.0.0.1:40000", token="tok"):
        self.base_url = base_url
        self.token = token


@pytest.fixture(autouse=True)
def _restore_state():
    """The module keeps process-wide breaker state; restore it per test."""
    saved = ai_cursor._FAILURES
    yield
    ai_cursor._FAILURES = saved


@pytest.fixture
def session(monkeypatch):
    session = FakeSession()
    manager = type("M", (), {"session": lambda self, serial: session})()
    monkeypatch.setattr(
        "artemis.runtime.helper_manager.helper_manager", manager, raising=False
    )
    return session


def test_disabled_setting_skips_entirely(monkeypatch, session):
    monkeypatch.setattr(settings, "ARTEMIS_AI_CURSOR", False)
    calls = []
    monkeypatch.setattr(ai_cursor, "_send_show", lambda *a: calls.append(a))
    ai_cursor.fire_ai_cursor("dev", 10, 20)
    ai_cursor._EXECUTOR.submit(lambda: None).result()  # drain the queue
    assert calls == []


def test_send_show_posts_cursor_command(monkeypatch, session):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["request"] = request
        captured["url"] = request.full_url
        captured["body"] = json.loads(request.data.decode())

        class Resp:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b"{}"

        return Resp()

    monkeypatch.setattr(ai_cursor.urllib.request, "urlopen", fake_urlopen)
    ai_cursor._send_show("dev", 540.0, 1080.0, 1100)
    assert captured["url"] == "http://127.0.0.1:40000/action"
    assert captured["body"]["cmd"] == "show_cursor"
    assert captured["body"]["x"] == 540.0 and captured["body"]["y"] == 1080.0
    # urllib stores header names capitalized; the token must ride along either way.
    assert captured["request"].headers.get("X-artemis-token") == "tok"


def test_breaker_stops_after_repeated_failures(monkeypatch, session):
    def boom(request, timeout=None):
        raise urllib.error.URLError("refused")

    monkeypatch.setattr(ai_cursor.urllib.request, "urlopen", boom)
    attempts = []
    real_open = boom
    monkeypatch.setattr(
        ai_cursor, "_MAX_CONSECUTIVE_FAILURES", 2
    )
    for _ in range(5):
        ai_cursor._send_show("dev", 1, 2, 100)
        attempts.append(ai_cursor._FAILURES)
    assert attempts == [1, 2, 2, 2, 2]  # breaker latches at the limit


def test_success_resets_failure_counter(monkeypatch, session):
    state = {"fail_first": True}

    def flaky(request, timeout=None):
        if state["fail_first"]:
            state["fail_first"] = False
            raise urllib.error.URLError("once")

        class Resp:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return b"{}"

        return Resp()

    monkeypatch.setattr(ai_cursor.urllib.request, "urlopen", flaky)
    ai_cursor._send_show("dev", 1, 2, 100)
    assert ai_cursor._FAILURES == 1
    ai_cursor._send_show("dev", 1, 2, 100)
    assert ai_cursor._FAILURES == 0


def test_missing_session_is_silent(monkeypatch):
    manager = type("M", (), {"session": lambda self, serial: None})()
    monkeypatch.setattr(
        "artemis.runtime.helper_manager.helper_manager", manager, raising=False
    )
    opened = []
    monkeypatch.setattr(
        ai_cursor.urllib.request, "urlopen", lambda req, timeout=None: opened.append(req)
    )
    ai_cursor._send_show("dev", 1, 2, 100)
    assert opened == []
    assert ai_cursor._FAILURES == 0  # not the helper's fault; not a failure
