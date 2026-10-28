@echo off
title Laya — Voice Assistant for Android
cd /d "voice assistant for android" 2>nul || (echo Run from D:\artemis-voice& pause& exit /b 1)
".venv\Scripts\pythonw.exe" voice_assistant.py
if errorlevel 1 (".venv\Scripts\python.exe" voice_assistant.py & pause)
