@echo off
title Laya Voice Assistant - Setup
echo ==========================================
echo  Laya Voice Assistant - one-time setup
echo ==========================================
cd /d "voice assistant for android" 2>nul || (echo Run from D:\artemis-voice& pause& exit /b 1)

REM 1) Independent venv (never touches D:\artemis\.venv or MCP)
python -m venv .venv || (echo Python 3.10+ required & pause & exit /b 1)

echo Installing packages (sounddevice, vosk, edge-tts, pygame, requests)...
.venv\Scripts\python -m pip install --quiet --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt || pause

echo Downloading the streaming speech model (~45 MB, one time)...
.venv\Scripts\python tools\download_model.py

echo.
echo [DONE] Setup complete. Launch with run_voice_assistant.bat
pause
