@echo off
chcp 65001 > nul
title Aria-FastFlow Desktop Studio (Apex v1.0)
echo ===============================================================================
echo   Aria-FastFlow: Ultra-Fast Desktop AI Assistant Studio
echo   (0KV Smart-KV x Flow-Matching 2-Step x 160-490 tok/s x TTFT 80ms)
echo ===============================================================================
echo.

set PYTHON_ENV=H:\DiffusionQwen_env\Scripts\python.exe

if not exist "%PYTHON_ENV%" (
    echo [!] H:\DiffusionQwen_env not found. Using default python...
    set PYTHON_ENV=python
)

echo [1/2] Preparing environment...
cd /d "%~dp0"

echo [2/2] Launching Aria-FastFlow CLI...
"%PYTHON_ENV%" chat_fastflow_apex_cli.py

echo.
echo Aria: Thank you Master! See you again!
pause
