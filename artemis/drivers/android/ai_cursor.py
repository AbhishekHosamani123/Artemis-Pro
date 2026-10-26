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

"""On-device "AI cursor" — a glowing blue pointer with an ``AI`` tag that the
Artemis Accessibility Helper draws (``TYPE_ACCESSIBILITY_OVERLAY``) at every
point the agent is about to touch, so a human watching the phone can follow
each move and tap — no matter whether the task came from the CLI, the web
console, or an IDE over MCP.

Fire-and-forget by contract: drawing happens on a one-slot background thread,
never adds latency to the action path, and is skipped entirely when the helper
is not attached (e.g. the UIAutomator2 hierarchy backend) or after repeated
failures. Disable with ``ARTEMIS_AI_CURSOR=false``.
"""

from __future__ import annotations

import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor

from third_party.mobile_use.utils.logger import get_logger

logger = get_logger(__name__)

#: One worker keeps cursor frames ordered; a queue depth of one draw is enough.
_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ai-cursor")

#: Stop trying after this many consecutive failures until one draw succeeds.
_MAX_CONSECUTIVE_FAILURES = 3

_FAILURES = 0


def ai_cursor_enabled() -> bool:
    """``ARTEMIS_AI_CURSOR`` (default true)."""
    try:
        from artemis.config import settings

        return bool(getattr(settings, "ARTEMIS_AI_CURSOR", True))
    except Exception:  # pragma: no cover - settings always importable in practice
        return True


def fire_ai_cursor(device_id: str, x: float, y: float, duration_ms: int = 1100) -> None:
    """Ask the helper to show the AI cursor at device pixel ``(x, y)``.

    Returns immediately; errors never propagate to the caller (the tap must
    not depend on the cosmetic overlay in any way).
    """
    if not ai_cursor_enabled():
        return
    try:
        _EXECUTOR.submit(_send_show, device_id, float(x), float(y), int(duration_ms))
    except RuntimeError:  # interpreter shutdown
        pass


def _send_show(device_id: str, x: float, y: float, duration_ms: int) -> None:
    global _FAILURES  # pylint: disable=global-statement
    if _FAILURES >= _MAX_CONSECUTIVE_FAILURES:
        return
    try:
        from artemis.runtime.helper_manager import helper_manager

        session = helper_manager.session(device_id)
        if session is None:
            # Helper backend not attached (uiautomator2 fallback or observer):
            # there is no overlay to draw with.
            return
        token = session.token or ""
        payload = json.dumps(
            {"cmd": "show_cursor", "x": x, "y": y, "duration": duration_ms, "token": token}
        ).encode("utf-8")
        request = urllib.request.Request(
            f"{session.base_url}/action",
            data=payload,
            method="POST",
            headers={"Content-Type": "application/json", "X-Artemis-Token": token},
        )
        with urllib.request.urlopen(request, timeout=1.5) as response:
            response.read()
        _FAILURES = 0
    except Exception as exc:  # pylint: disable=broad-exception-caught
        _FAILURES += 1
        if _FAILURES >= _MAX_CONSECUTIVE_FAILURES:
            logger.warning(
                f"AI cursor overlay disabled for now after {_FAILURES} failed attempts: {exc}"
            )
        else:
            logger.debug(f"AI cursor draw failed ({exc}); will retry.")
