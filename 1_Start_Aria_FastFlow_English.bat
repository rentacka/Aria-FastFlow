@echo off
chcp 65001 > nul
title Aria-FastFlow English Desktop Studio (Apex v1.0)
echo ===============================================================================
echo   Aria-FastFlow: Ultra-Fast English AI Assistant Studio
echo   (0KV Smart-KV x English Flow-Matching x 350 tok/s x Kaomoji Tail-Stitcher)
echo ===============================================================================
echo.

set PYTHON_ENV=H:\DiffusionQwen_env\Scripts\python.exe

if not exist "%PYTHON_ENV%" (
    echo [!] H:\DiffusionQwen_env not found. Using default python...
    set PYTHON_ENV=python
)

cd /d "%~dp0"

echo [1/1] Launching Aria-FastFlow English Studio...
echo.
"%PYTHON_ENV%" chat_fastflow_english_cli.py

echo.
echo Thank you for using Aria-FastFlow!
pause
