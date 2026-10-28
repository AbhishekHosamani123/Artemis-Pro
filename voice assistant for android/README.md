# 🎙️ Laya — Voice Assistant for Android  (cute buddy edition)

Talk to your phone through your PC microphone. This lives in a **separate folder**
(`D:\artemis-voice\voice assistant for android`) and runs in its **own venv**, so your
original `D:\artemis` install, its MCP integration in OpenCode, and all existing tools
keep working exactly as before.

## What it does
| You say | It does |
|---|---|
| "open camera" / "take a photo of me" | Camera opens instantly (⚡ sub-second ADB launch) |
| "open LinkedIn" / "open YouTube" | App opens instantly ⚡ |
| "go home" | Home screen instantly ⚡ |
| "open LinkedIn and search AI engineer jobs" | The Artemis agent does the whole flow on your phone (typing, tapping) |
| "stop" | Cancels the running agent task |
| Anytime it needs your decision | It **asks you**, with the classic **triangle arrow cursor** pointing at its question 🖱️ |

Speech is transcribed **as words appear** (live partial results) and the command fires
the moment you pause — no buttons, no waiting.

Replies come in a **cute neural voice** ("Done! What do you want to do next?"),
and the little marshmallow buddy waves, blinks, and wiggles while talking.

## Setup (one time)
1. Double-click `setup.bat` — creates its own venv, installs deps, downloads the
   ~45 MB streaming speech model (Indian-English flavored, US fallback).
2. Double-click `run.bat` — the window appears, says hi, and starts listening.

## Extras
- `python voice_assistant.py --speech` — test the cute voice only
- `python voice_assistant.py --text "open camera"` — run one command without mic (testing)
- `python voice_assistant.py --mic-list` — list microphones
- `config.json` — change TTS voice, mic mute behavior, task profile, serial override

## How it talks to your phone (no MCP impact)
- **Instant path:** raw ADB → launches apps/home/camera directly on your Redmi.
- **Agent path:** POSTs to the Artemis daemon REST API (`/api/run` → watches
  `/api/sessions/<id>`) — the **same pipeline** the MCP server uses. If the daemon
  isn't running yet, it starts it with the main install's venv — read-only, never
  modified. Nothing in `D:\artemis` is changed.

## Handmade icons (no external assets)
The buddy, arrow cursor and bubble are all drawn on a Tk canvas in
`voice_assistant.py` (`AssistantUI._draw_static`) — you can recolor or reshape them there.
