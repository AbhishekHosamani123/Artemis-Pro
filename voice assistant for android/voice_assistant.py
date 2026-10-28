# -*- coding: utf-8 -*-
"""
Laya Voice Assistant for Android
================================
Talk to your phone! A cute, always-on-top desktop companion that:
  * Streams your speech to text AS-you-speak (real-time partials)
  * Fires instant phone actions the moment you pause (open camera / apps / home)
  * Hands complex commands ("search AI engineer jobs on LinkedIn") to the
    Artemis daemon agent queue (same pipeline MCP uses => MCP unaffected)
  * Replies in a cute, natural voice (Edge neural TTS, cheerful)
  * Shows a cute marshmallow-style 2D buddy; when it needs your decision it
    points at the question bubble with a classic triangle arrow cursor.

Run:    python voice_assistant.py             (full voice assistant GUI)
        python voice_assistant.py --text "open camera"
        python voice_assistant.py --speech    (cute-voice self test)
        python voice_assistant.py --mic-list  (list microphones)

This app lives beside the main Artemis install and only *borrows* its venv
read-only to spawn the daemon if MCP hasn't already started one — the main
install and its MCP integration are never modified.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import queue
import random
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_DIR = HERE / "models"
CONFIG_PATH = HERE / "config.json"

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
DAEMON_PORT_DEFAULT = 8000
VOSK_MODEL_URLS = [
    "https://alphacephei.com/vosk/models/vosk-model-small-en-in-0.4.zip",   # Indian English
    "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",  # fallback
]
ADB_CANDIDATES = [
    Path(os.environ.get("LOCALAPPDATA", "")) / "Android" / "Sdk" / "platform-tools" / "adb.exe",
    Path("D:/artemis/.venv/Scripts/adb.exe"),
]
ARTEMIS_VENV_PY = Path("D:/artemis/.venv/Scripts/python.exe")
ARTEMIS_ROOT = Path("D:/artemis")

DEFAULT_CONFIG = {
    "tts_voice": "en-IN-NeerjaNeural",      # cute, lively Indian-English female neural voice
    "tts_voice_backup": "en-US-AnaNeural",  # child-like cheerful voice
    "device_serial": None,                  # auto-detect
    "task_profile": "flash",
    "mic_mute_while_speaking": True,
    "daemon_port": 8000,
    "auto_start_daemon": True,
}


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
        except Exception:
            pass
    return cfg


def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------
# ADB helpers
# --------------------------------------------------------------------------
def find_adb() -> Path | None:
    for cand in ADB_CANDIDATES:
        try:
            if cand and cand.exists():
                return cand
        except Exception:
            continue
    which = shutil.which("adb")
    return Path(which) if which else None


def run_adb(adb: Path, args: list[str], timeout: int = 10, serial: str | None = None) -> str:
    cmd = [str(adb)]
    if serial:
        cmd += ["-s", serial]
    cmd += args
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                         creationflags=subprocess.CREATE_NO_WINDOW)
    return (out.stdout or "")


def list_devices(adb: Path) -> list[str]:
    try:
        txt = run_adb(adb, ["devices"], timeout=6)
        return [ln.split()[0] for ln in txt.splitlines()[1:]
                if len(ln.split()) >= 2 and ln.split()[1] == "device"]
    except Exception:
        return []


def ensure_vosk_model() -> Path | None:
    """Download the small streaming speech model once (Indian English preferred)."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name in ("vosk-model-small-en-in-0.4", "vosk-model-small-en-us-0.15"):
        p = MODEL_DIR / name
        if p.exists() and (p / "conf" / "model.conf").exists():
            return p
    for url in VOSK_MODEL_URLS:
        zip_path = MODEL_DIR / url.split("/")[-1]
        try:
            print(f"⬇ Downloading speech model {zip_path.name} (~45 MB, one time)…", flush=True)
            urllib.request.urlretrieve(url, zip_path)
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(MODEL_DIR)
            zip_path.unlink(missing_ok=True)
            folder = MODEL_DIR / zip_path.stem
            if folder.exists():
                return folder
        except Exception as e:  # noqa: BLE001
            print(f"  model download failed ({url}): {e}", flush=True)
    return None


# --------------------------------------------------------------------------
# Daemon bridge (same REST contract used by the MCP server)
# --------------------------------------------------------------------------
class DaemonBridge:
    def __init__(self, port: int = 8000, auto_start: bool = True):
        self.base = f"http://127.0.0.1:{port}"
        self.port = port
        self.auto_start = auto_start

    def is_running(self, timeout: float = 3.0) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base}/api/status", timeout=timeout):
                return True
        except Exception:
            return False

    def ensure_running(self, wait_seconds: float = 90.0) -> bool:
        if self.is_running():
            return True
        if not self.auto_start:
            return False
        py, root = ARTEMIS_VENV_PY, ARTEMIS_ROOT
        if not (py.exists() and (root / "apps" / "admin_console" / "server.py").exists()):
            print("❌ Cannot auto-start daemon: main Artemis venv not found at", py, flush=True)
            return False
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONUTF8"] = "1"
        cmd = [str(py), "-m", "apps.admin_console.server", "--host", "127.0.0.1", "--port", str(self.port)]
        print("🚀 Starting Artemis daemon (same engine the MCP server uses)…", flush=True)
        kwargs = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = (
                subprocess.CREATE_NEW_PROCESS_GROUP
                | getattr(subprocess, "DETACHED_PROCESS", 0x00000008)
                | getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
            )
        try:
            subprocess.Popen(cmd, cwd=str(root), env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                             stdin=subprocess.DEVNULL, **kwargs)
        except Exception as e:  # noqa: BLE001
            print(f"  daemon spawn failed: {e}", flush=True)
            return False
        deadline = time.monotonic() + wait_seconds
        while time.monotonic() < deadline:
            if self.is_running(1.5):
                print("✓ Daemon ready.", flush=True)
                return True
            time.sleep(1.0)
        return self.is_running(3.0)

    def submit(self, goal: str, device_serial: str | None = None,
               profile: str = "flash", locked_app_package: str | None = None) -> dict | None:
        payload = {
            "goal": goal, "profile": profile, "device_serial": device_serial,
            "verification_level": "final", "explorer_mode": "flash",
            "locked_app_package": locked_app_package, "ingress": "voice-assistant",
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.base}/api/run", data=data,
            headers={"Content-Type": "application/json", "User-Agent": "Artemis-Voice-Assistant"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=25) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = ""
            try:
                body = e.read().decode()[:300]
            except Exception:
                pass
            print(f"  /api/run error {e.code}: {body}", flush=True)
            return None
        except Exception as e:  # noqa: BLE001
            print(f"  /api/run failed: {e}", flush=True)
            return None

    def session(self, session_id: str, timeout: float = 6.0) -> dict | None:
        try:
            with urllib.request.urlopen(
                    f"{self.base}/api/sessions/{session_id}", timeout=timeout) as r:
                return json.loads(r.read().decode())
        except Exception:
            return None

    def stop(self, session_id: str) -> bool:
        data = json.dumps({"session_id": session_id}).encode()
        req = urllib.request.Request(
            f"{self.base}/api/stop", data=data,
            headers={"Content-Type": "application/json", "User-Agent": "Artemis-Voice-Assistant"},
            method="POST")
        try:
            with urllib.request.urlopen(req, timeout=6) as r:
                return json.loads(r.read().decode()).get("status") == "stopped"
        except Exception:
            return False


class TaskWatcher(threading.Thread):
    """Polls a submitted daemon session; reports completion/failure/progress."""

    def __init__(self, daemon: DaemonBridge, session_id: str,
                 on_done, on_fail, on_progress=None):
        super().__init__(daemon=True)
        self.daemon, self.sid = daemon, session_id
        self.on_done, self.on_fail = on_done, on_fail
        self.on_progress = on_progress or (lambda *_: None)

    def run(self) -> None:
        start = time.monotonic()
        last_status = None
        while time.monotonic() - start < 420:  # 7 min cap
            info = self.daemon.session(self.sid)
            if info:
                status = (info.get("status") or info.get("state") or "").lower()
                if status and status != last_status:
                    self.on_progress(status)
                    last_status = status
                if status in ("completed", "success", "passed", "finished"):
                    result = info.get("result") or info.get("final_answer") or ""
                    self.on_done(result)
                    return
                if status in ("failed", "error", "stopped", "cancelled", "canceled"):
                    err = info.get("error") or info.get("last_error") or "the task didn't finish"
                    self.on_fail(str(err))
                    return
            time.sleep(2.0)
        self.on_fail("timed out waiting for the task to finish")


# --------------------------------------------------------------------------
# Instant fast-path router (pure ADB, no LLM — sub-second actions)
# --------------------------------------------------------------------------
APP_HINTS = {
    "camera": "com.android.camera",
    "chrome": "com.android.chrome",
    "youtube": "com.google.android.youtube",
    "linkedin": "com.linkedin.android",
    "instagram": "com.instagram.android",
    "whatsapp": "com.whatsapp",
    "gmail": "com.google.android.gm",
    "telegram": "org.telegram.messenger",
    "spotify": "com.spotify.music",
    "settings": "com.android.settings",
    "play store": "com.android.vending",
    "flipkart": "com.flipkart.android",
    "amazon": "in.amazon.mShop.android.shopping",
    "snapchat": "com.snapchat.android",
    "facebook": "com.facebook.katana",
    "google maps": "com.google.android.apps.maps",
    "maps": "com.google.android.apps.maps",
    "drive": "com.google.android.apps.docs",
}

RE_CAMERA = re.compile(
    r"\b(take|capture|click|snap)\s+(?:a\s+)?(photo|pic|picture|selfie)\b", re.I)
RE_OPEN = re.compile(r"^(?:ok\s*)?(?:open|launch|start|go\s+to)\s+([a-z0-9 .\-']+)$", re.I)
RE_HOME = re.compile(r"\bhome\s*screen$|^go\s*home$|^home$", re.I)
RE_STOP = re.compile(r"^(?:stop|cancel|abort)(?:\s+(?:it|task|that))?$", re.I)


class FastRouter:
    def __init__(self, adb: Path | None, serial: str | None, ui):
        self.adb, self.serial, self.ui = adb, serial, ui
        self._packages: list[str] | None = None

    def refresh_packages(self) -> None:
        if not self.adb:
            return
        try:
            txt = run_adb(self.adb, ["shell", "pm", "list", "packages"], timeout=15, serial=self.serial)
            self._packages = [ln.replace("package:", "").strip()
                              for ln in txt.splitlines() if ln.startswith("package:")]
            print(f"  cached {len(self._packages)} device packages for instant launch", flush=True)
        except Exception:
            self._packages = None

    def _resolve(self, spoken: str) -> str | None:
        name = spoken.lower().strip().strip(".").replace("please", "").strip()
        for hint, pkg in APP_HINTS.items():
            if hint in name:
                return pkg
        if self._packages:
            in_name = name.replace(" ", "")
            for pkg in self._packages:
                short = pkg.split(".")[-1].lower()
                if in_name and (in_name in short or short in in_name or in_name in pkg):
                    return pkg
        return None

    def try_fast(self, text: str) -> bool:
        """Handle instantly with pure ADB. Returns True when handled."""
        m = RE_HOME.match(text.strip())
        if m and self.adb and self.serial:
            try:
                run_adb(self.adb, ["shell", "input", "keyevent", "KEYCODE_HOME"],
                        timeout=8, serial=self.serial)
                self.ui.reply("🏠 Home screen! What do you want to do next?", instant=True)
                return True
            except Exception as e:  # noqa: BLE001
                print(f"fast home failed: {e}", flush=True)

        m = RE_CAMERA.search(text)
        if m and self.adb and self.serial:
            try:
                run_adb(self.adb, ["shell", "am", "start", "-a", "android.media.action.STILL_CAPTURE_IMAGE"],
                        timeout=8, serial=self.serial)
                self.ui.reply("📸 Camera's open! Just tap the big circle to snap your photo — "
                              "what do you want to do next?", instant=True)
                return True
            except Exception as e:  # noqa: BLE001
                print(f"camera fast-path failed: {e}", flush=True)
                return False

        m = RE_OPEN.match(text.strip())
        if m and self.adb and self.serial:
            pkg = self._resolve(m.group(1))
            if pkg:
                try:
                    run_adb(self.adb, ["shell", "monkey", "-p", pkg,
                                       "-c", "android.intent.category.LAUNCHER", "1"],
                            timeout=12, serial=self.serial)
                    app_name = m.group(1).strip().title()
                    self.ui.reply(f"⚡ {app_name} is open! What do you want to do next?", instant=True)
                    return True
                except Exception as e:  # noqa: BLE001
                    print(f"open fast-path failed: {e}", flush=True)
        return False


# --------------------------------------------------------------------------
# Cute speech (Edge-TTS neural voices; pyttsx3 offline fallback)
# --------------------------------------------------------------------------
class CuteVoice:
    def __init__(self, cfg: dict):
        self.voice = cfg.get("tts_voice", "en-IN-NeerjaNeural")
        self.backup = cfg.get("tts_voice_backup", "en-US-AnaNeural")
        self.busy = threading.Lock()
        self.on_talk_start = lambda: None
        self.on_talk_end = lambda: None

    def _speak_sync(self, text: str) -> None:
        import edge_tts
        out = HERE / "tts_cache" / "reply.mp3"
        out.parent.mkdir(exist_ok=True)
        for v in (self.voice, self.backup):
            try:
                com = edge_tts.Communicate(text, v)
                asyncio.run(com.save(str(out)))
                if out.exists() and out.stat().st_size > 500:
                    self._play(out)
                    return
            except Exception:
                continue
        self._speak_offline(text)

    @staticmethod
    def _play(mp3: Path) -> None:
        import pygame
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(str(mp3))
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            time.sleep(0.05)

    @staticmethod
    def _speak_offline(text: str) -> None:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception:
            pass  # stay silent rather than crash

    def say(self, text: str) -> None:
        def worker():
            with self.busy:
                self.on_talk_start()
                try:
                    self._speak_sync(text)
                except Exception as e:  # noqa: BLE001
                    print(f"TTS failed: {e}", flush=True)
                    self._speak_offline(text)
                finally:
                    self.on_talk_end()
        threading.Thread(target=worker, daemon=True).start()


# --------------------------------------------------------------------------
# Streaming recognizer (Vosk) — transcribes as words appear, fires on pause
# --------------------------------------------------------------------------
class Ears(threading.Thread):
    def __init__(self, on_partial, on_final, mute_flag: threading.Event,
                 model_dir: Path | None):
        super().__init__(daemon=True)
        self.on_partial, self.on_final = on_partial, on_final
        self.mute_flag = mute_flag
        self.model_path = model_dir
        self.want_stop = False

    def run(self) -> None:
        try:
            import sounddevice as sd
            from vosk import Model, KaldiRecognizer, SetLogLevel
            SetLogLevel(-1)
            if not self.model_path:
                self.on_final("", True)
                self.on_final("__nomodel__", True)
                return
            model = Model(str(self.model_path))
            rec = KaldiRecognizer(model, 16000)
            stream = sd.RawInputStream(samplerate=16000, blocksize=4000,
                                       dtype="int16", channels=1)
            stream.start()
            print("👂 mic running (16 kHz mono) - listening for your voice", flush=True)
            while not self.want_stop:
                if self.mute_flag.is_set():
                    try:
                        rec.Reset()
                    except Exception:
                        pass
                    time.sleep(0.08)
                    continue
                data, _ = stream.read(4000, exception_on_overflow=False)
                buf = bytes(data)
                if rec.AcceptWaveform(buf):
                    text = (json.loads(rec.FinalResult()).get("text") or "").strip()
                    if text:
                        self.on_final(text)
                    else:
                        self.on_partial("")
                else:
                    partial = (json.loads(rec.PartialResult()).get("partial") or "").strip()
                    self.on_partial(partial)
            stream.stop()
            stream.close()
        except Exception as e:  # noqa: BLE001
            print(f"🎧 Mic error: {e}", flush=True)
            self.on_final(f"__micerr__ {e}", True)


# --------------------------------------------------------------------------
# The cute UI (animated marshmallow buddy + triangle arrow cursor)
# --------------------------------------------------------------------------
BG = "#101010"
BODY = "#F3EAD7"
FACE = "#FAF3E6"
EYE = "#211D17"
BLUSH = "#F0C4C8"
MOUTH = "#7A5140"
BUBBLE = "#FFFDF7"


class AssistantUI:
    def __init__(self, cfg: dict):
        import tkinter as tk
        self.tk = tk
        self.cfg = cfg
        self.root = tk.Tk()
        self.root.title("Laya — Voice Assistant for Android")
        self.root.configure(bg=BG)
        self.root.attributes("-topmost", True)
        self.root.geometry("460x680+1200+80")
        self.root.config(cursor="arrow")   # classic triangle pointer, like a PC
        W, H = 460, 680
        self.canvas = tk.Canvas(self.root, width=W, height=H, bg=BG, highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.W, self.H = W, H

        self.say_queue: "queue.Queue[str]" = queue.Queue()
        self.ui_queue: "queue.Queue[tuple]" = queue.Queue()
        self.talking = False
        self.blink_until = time.monotonic() + random.uniform(2.0, 4.0)
        self.eye_hidden = False
        self.wave_phase = 0.0
        self.bob_phase = 0.0
        self.arrow_id = None
        self.arrow_visible = False
        self.arrow_phase = 0.0
        self._ids: dict[str, int] = {}

        self._draw_static()
        self._tick()
        self._process_queues()

    # ---------------- canvas helpers ----------------
    def _draw_static(self) -> None:
        c, W = self.canvas, self.W
        c.create_oval(150, 600, 310, 626, fill="#1c1c1c", outline="")  # ground shadow
        parts = []
        parts.append(c.create_oval(160, 330, 300, 596, fill=BODY, outline=""))          # body
        parts.append(c.create_oval(150, 186, 310, 430, fill=BODY, outline=""))          # hood dome
        parts.append(c.create_oval(197, 252, 263, 356, fill=FACE, outline=""))          # face patch
        parts.append(c.create_line(294, 356, 347, 306, smooth=True, width=26,
                                   fill=BODY, capstyle="round", tags="arm"))           # waving arm
        parts.append(c.create_oval(322, 277, 362, 317, fill=BODY, outline="", tags="armhand"))
        parts.append(c.create_line(168, 366, 138, 428, smooth=True, width=24,
                                   fill=BODY, capstyle="round"))                       # left arm
        parts.append(c.create_oval(118, 414, 158, 446, fill=BODY, outline=""))
        parts.append(c.create_oval(196, 580, 236, 606, fill=BODY, outline=""))          # feet
        parts.append(c.create_oval(228, 580, 268, 606, fill=BODY, outline=""))
        self._ids["parts"] = parts
        # face bits
        self._ids["eyeL"] = c.create_oval(214, 288, 225, 300, fill=EYE, outline="")
        self._ids["eyeR"] = c.create_oval(237, 288, 248, 300, fill=EYE, outline="")
        self._ids["gL"] = c.create_oval(216, 290, 219, 293, fill="#fff", outline="")
        self._ids["gR"] = c.create_oval(239, 290, 242, 293, fill="#fff", outline="")
        self._ids["blushL"] = c.create_oval(204, 310, 217, 319, fill=BLUSH, outline="")
        self._ids["blushR"] = c.create_oval(244, 310, 257, 319, fill=BLUSH, outline="")
        self._ids["mouth"] = c.create_arc(220, 306, 244, 322, start=205, extent=130,
                                          style="arc", outline=MOUTH, width=3)
        # speech bubble
        c.create_oval(38, 26, 422, 186, fill=BUBBLE, outline="#e7dcc4", width=2)
        tail = c.create_polygon(213, 184, 231, 214, 247, 184, fill=BUBBLE, outline="#e7dcc4")
        c.tag_lower(tail)
        self._ids["bubble_text"] = c.create_text(
            230, 104,
            text=("👋 Hi! I'm Laya, your cute assistant.\n"
                  "Just say things like: 'open camera',\n"
                  "'open LinkedIn', or 'open LinkedIn and\n"
                  "search AI engineer jobs'.\nDon't forget to say 'Done' 😉"),
            font=("Segoe UI", 11, "bold"), width=350, justify="center")
        # mic level strip + status
        self._ids["micbar"] = c.create_rectangle(40, 620, 160, 630, fill="#1d1d1d", outline="#333")
        self._ids["status"] = c.create_text(330, 625, text="✨ booting up…",
                                            font=("Segoe UI", 10), fill="#9ad07f")

    # ---------------- animation ----------------
    def _tick(self) -> None:
        t = time.monotonic()
        self.wave_phase += 0.16
        self.bob_phase += 0.05
        bob = math.sin(self.bob_phase) * 2.5
        sway = math.sin(self.bob_phase) * 1.2
        swing = math.sin(self.wave_phase) * 16
        self.canvas.coords("arm", 294 + sway, 356 + bob, 347 + swing * 0.3, 306 - abs(swing) * 0.65 + bob)
        self.canvas.coords("armhand", 322 + swing * 0.35, 277 - abs(swing) * 0.8 + bob,
                           362 + swing * 0.35, 317 - abs(swing) * 0.8 + bob)
        # blink
        if t > self.blink_until and not self.eye_hidden:
            self.blink_until = t + 0.15
            self.eye_hidden = True
            for k in ("eyeL", "eyeR", "gL", "gR"):
                self.canvas.itemconfigure(self._ids[k], state="hidden")
        elif t > self.blink_until and self.eye_hidden:
            self.blink_until = t + random.uniform(2.4, 5.2)
            self.eye_hidden = False
            for k in ("eyeL", "eyeR", "gL", "gR"):
                self.canvas.itemconfigure(self._ids[k], state="normal")
        # mouth: talking bobble vs gentle smile
        self.canvas.delete("mO")
        if self.talking and int(t * 10) % 2:
            self.canvas.create_oval(224, 307, 240, 320, fill=MOUTH, outline="", tags="mO")
            self.canvas.itemconfigure(self._ids["mouth"], state="hidden")
        else:
            self.canvas.itemconfigure(self._ids["mouth"], state="normal")
        # triangle arrow cursor (desktop-PC style pointer)
        if self.arrow_visible:
            self.arrow_phase += 0.25
            wig = int(math.sin(self.arrow_phase) * 7)
            tip_x, tip_y = (206 + wig, 196 + bob)
            pts = [tip_x, tip_y, tip_x, tip_y + 32, tip_x + 12, tip_y + 22,
                   tip_x + 22, tip_y + 42, tip_x + 31, tip_y + 38, tip_x + 21,
                   tip_y + 19, tip_x + 33, tip_y + 15]
            if self.arrow_id is None:
                self.arrow_id = self.canvas.create_polygon(
                    pts, fill="#ffffff", outline="#222", width=2, tags="arrowcur")
            else:
                self.canvas.coords(self.arrow_id, *pts)
        elif self.arrow_id is not None:
            self.canvas.delete("arrowcur")
            self.arrow_id = None
        self.root.after(60, self._tick)

    # ---------------- public API (thread-safe via queue) ----------------
    def reply(self, text: str, instant: bool = False, needs_help: bool = False) -> None:
        self.ui_queue.put(("bubble", text, needs_help))
        self.say_queue.put(text)

    def live_partial(self, text: str) -> None:
        self.ui_queue.put(("partial", text))

    def status(self, text: str) -> None:
        self.ui_queue.put(("status", text))

    def talk(self, on: bool) -> None:
        self.ui_queue.put(("talking", on))

    def _process_queues(self) -> None:
        try:
            while True:
                ev = self.ui_queue.get_nowait()
                kind = ev[0]
                if kind == "bubble":
                    self.canvas.itemconfigure(self._ids["bubble_text"], text="🧚 " + ev[1])
                    self.arrow_visible = bool(ev[2])
                    if self.arrow_visible:
                        self.arrow_phase = 0.0
                elif kind == "partial":
                    if ev[1]:
                        self.canvas.itemconfigure(self._ids["bubble_text"], text="🎤 you: " + ev[1])
                        self.arrow_visible = False
                elif kind == "status":
                    self.canvas.itemconfigure(self._ids["status"], text=ev[1])
                elif kind == "talking":
                    self.talking = bool(ev[1])
        except queue.Empty:
            pass
        try:
            while True:
                text = self.say_queue.get_nowait()
                self.voice_impl.say(text)
        except queue.Empty:
            pass
        self.root.after(120, self._process_queues)

    def bootstrap(self, voice: CuteVoice, mute_flag: threading.Event) -> None:
        self.voice_impl = voice

        def start_talk():
            self.talk(True)
            mute_flag.set()          # don't listen to own cute voice

        def end_talk():
            self.talk(False)
            mute_flag.clear()
        voice.on_talk_start = start_talk
        voice.on_talk_end = end_talk


# --------------------------------------------------------------------------
# Orchestrator: ears → router / daemon → cute voice + GUI
# --------------------------------------------------------------------------
class Assistant:
    def __init__(self, cfg: dict, text_mode: str | None):
        self.cfg = cfg
        self.text_mode = text_mode
        self.ui = AssistantUI(cfg)
        self.voice = CuteVoice(cfg)
        self.mute_flag = threading.Event()
        self.ui.bootstrap(self.voice, self.mute_flag)
        self.daemon = DaemonBridge(port=int(cfg.get("daemon_port", 8000)),
                                   auto_start=cfg.get("auto_start_daemon", True))
        self.adb = find_adb()
        self.serial: str | None = cfg.get("device_serial")
        self.router = FastRouter(self.adb, self.serial, self.ui)
        self.current_sid: str | None = None
        self.busy_task = False
        self.ready = False

    # ---- lifecycle ----
    def start(self) -> None:
        threading.Thread(target=self._boot, daemon=True).start()
        if self.text_mode:
            threading.Thread(target=self._text_pipeline, daemon=True).start()
        else:
            self._start_mic()
        self.ui.root.mainloop()

    def _boot(self) -> None:
        self.ui.status("⏳ checking Artemis daemon…")
        ok = self.daemon.ensure_running()
        dev_note = ""
        if self.adb:
            devs = list_devices(self.adb)
            if devs:
                self.serial = self.serial or devs[0]
                dev_note = f"   📱 {self.serial}"
            else:
                dev_note = "   ⚠ no phone detected"
        else:
            dev_note = "   ⚠ adb missing — only agent tasks will work"
        self.router.serial = self.serial
        self.router.refresh_packages()
        self.ready = ok or not self.cfg.get("auto_start_daemon", True)
        if ok:
            self.ui.status(f"🟢 Artemis ready{dev_note}")
        else:
            self.ready = False
            self.ui.status(f"🔴 daemon down{dev_note}")
        self.ui.ready = self.ready

    # ---- voice path ----
    def _start_mic(self) -> None:
        model = ensure_vosk_model()
        ears = Ears(on_partial=self._on_partial,
                    on_final=self._on_final,
                    mute_flag=self.mute_flag, model_dir=model)
        ears.start()
        threading.Thread(target=self.voice.say,
                         args=("Hi! I'm Laya! Tell me what to do — like open camera, "
                               "or open LinkedIn and search AI engineer jobs!",),
                         daemon=True).start()

    def _on_partial(self, text: str) -> None:
        self.ui.live_partial(text)

    def _on_final(self, text: str, is_error: bool = False) -> None:
        if is_error:
            if text.startswith("__nomodel__"):
                self.ui.reply("I couldn't download the speech model. "
                              "Please check internet and restart me!", needs_help=True)
            elif text.startswith("__micerr__"):
                self.ui.reply(f"I can't reach your microphone: {text[10:]}", needs_help=True)
            return
        if not text.strip():
            self.ui.live_partial("")
            return
        print(f"🗣 heard: {text}", flush=True)
        self.handle_command(text)

    # ---- command pipeline (voice and text modes share this) ----
    def handle_command(self, text: str) -> None:
        if not self.ready:
            self.ui.reply("Still booting! One moment, please…", instant=True)
            return
        if RE_STOP.match(text.strip()):
            if self.busy_task:
                self.request_stop()
            else:
                self.ui.reply("Nothing is running right now! What do you want to do next?",
                              instant=True)
            return
        if self.busy_task:
            self.ui.reply("I'm still working on your last one! Say 'stop' to cancel it, "
                          "or wait for my 'Done!'.", instant=True)
            return
        threading.Thread(target=self._process, args=(text.strip(),), daemon=True).start()

    def _process(self, text: str) -> None:
        if not text:
            return
        # 1) instant fast path (sub-second, no AI wait)
        if self.router.try_fast(text):
            return
        # 2) hand the full command to the Artemis agent via the daemon
        self.busy_task = True
        self.voice.say("Getting on it right away!")
        self.ui.reply(f"🤖 On it! '{text}' — working on your phone now…")
        self.ui.status(f"🤖 task running…   📱 {self.serial or '?'}")
        resp = self.daemon.submit(goal=text, device_serial=self.serial,
                                  profile=self.cfg.get("task_profile", "flash"))
        if not resp:
            self.busy_task = False
            self.ui.reply("Hmm, I couldn't queue that task. What would you like me to try instead?",
                          needs_help=True)
            return
        sid = resp.get("session_id") or resp.get("sessionId") or resp.get("id") or resp.get("trace_id")
        if not sid:
            self.busy_task = False
            self.ui.reply("I queued it but the answer looked odd — let's try again!", needs_help=True)
            return
        self.current_sid = sid
        TaskWatcher(self.daemon, sid, on_done=self._task_done, on_fail=self._task_fail,
                    on_progress=self._task_progress).start()

    def _task_progress(self, status: str) -> None:
        pretty = {"queued": "⏳ queued", "running": "🏃 running…"}
        self.ui.status(f"🤖 {pretty.get(status, status)}   📱 {self.serial or '?'}")

    def _task_done(self, result: str) -> None:
        self.busy_task = False
        short = (result or "").strip()
        if len(short) > 170:
            short = short[:167] + "…"
        msg = (f"Done! {short} — what do you want to do next?" if short
               else "Done! What do you want to do next?")
        self.ui.reply(msg)
        self.ui.status("🟢 idle   ✨ ready for next command")

    def _task_fail(self, err: str) -> None:
        self.busy_task = False
        clean = (err or "unexpected failure").strip().replace("\n", " ")
        if len(clean) > 190:
            clean = clean[:187] + "…"
        self.ui.reply(f"Oops, I hit a snag: {clean} — what should I do instead?", needs_help=True)
        self.ui.status("🟡 needs your help")

    def request_stop(self) -> None:
        sid = self.current_sid
        if sid:
            self.daemon.stop(sid)
            self.ui.reply("🛑 Stopped! What do you want to do next?", instant=True)
        else:
            self.ui.reply("Nothing is running! What do you want to do next?", instant=True)

    # ---- one-shot text mode (pipeline testing without mic) ----
    def _text_pipeline(self) -> None:
        cmd = self.text_mode or ""
        print(f"[text-mode] → {cmd!r}", flush=True)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            if self.ready:
                break
            time.sleep(1.0)
        if not self.ready:
            print("[text-mode] assistant never became ready; aborting.", flush=True)
            self.ui.root.after(0, self.ui.root.destroy)
            return
        if cmd == "__say__":
            self.ui.reply("Hello! I am Laya, your cute voice assistant. I am ready!")
        else:
            self.handle_command(cmd)
        # give TTS + task a moment, then close
        deadline = time.monotonic() + 540
        while time.monotonic() < deadline:
            if not self.busy_task:
                time.sleep(14)
                break
            time.sleep(2.0)
        self.ui.root.after(0, self.ui.root.destroy)


# --------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Laya voice assistant for your Android phone")
    p.add_argument("--text", default=None,
                   help='run one spoken-style command, e.g. --text "open camera"')
    p.add_argument("--speech", action="store_true", help="test the cute TTS voice and exit")
    p.add_argument("--ready", action="store_true", help="boot check + ready greeting, then exit")
    p.add_argument("--mic-list", action="store_true", help="list microphones and exit")
    p.add_argument("--device", default=None, help="adb serial override")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_config()

    if args.mic_list:
        try:
            import sounddevice as sd
            print(sd.query_devices())
            return 0
        except Exception as e:  # noqa: BLE001
            print(f"sounddevice issue: {e}")
            return 1

    missing = []
    for mod in ("vosk", "sounddevice", "edge_tts", "pygame"):
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    if missing:
        print(f"⚠ missing modules: {', '.join(missing)} — run setup.bat first.", flush=True)
        if args.speech:
            return 1

    if args.speech:
        voice = CuteVoice(cfg)
        voice.on_talk_start = lambda: print("🔊 speaking…", flush=True)
        voice.on_talk_end = lambda: print("🔊 done.", flush=True)
        voice.say("Hello! I am Laya, your cute voice assistant. Everything is ready — "
                  "what do you want to do next?")
        try:
            while True:
                time.sleep(0.3)
        except KeyboardInterrupt:
            return 0

    if args.device:
        cfg["device_serial"] = args.device
        save_config(cfg)

    if args.ready:
        assistant = Assistant(cfg, text_mode="__say__")   # greeting only, then exit
    else:
        assistant = Assistant(cfg, text_mode=args.text)
    assistant.start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
