@echo off
chcp 65001 > nul
title Aria-FastFlow English LoRA Apex Trainer (RTX 3060)
echo ===============================================================================
echo   Studio 34: Aria-FastFlow English LoRA Apex Trainer (RTX 3060 12GB)
echo   Master's Vision: UltraChat x Barron CE x English N-gram x FastFlow LoRA
echo ===============================================================================
echo.

set PYTHON_ENV=H:\DiffusionQwen_env\Scripts\python.exe

if not exist "%PYTHON_ENV%" (
    echo [!] H:\DiffusionQwen_env not found. Using default python...
    set PYTHON_ENV=python
)

cd /d "%~dp0"

echo [1/2] Checking English dataset...
if not exist "dataset_cache\aria_home_training_set_en.jsonl" (
    echo [*] Building English dataset...
    "%PYTHON_ENV%" build_english_home_dataset.py
)

echo.
echo [2/2] Training English LoRA (Target: 1,000 steps / ~15-20 min)...
echo.
"%PYTHON_ENV%" train_english_diffusion_gemma.py

echo.
echo Aria: English LoRA training completed successfully!
pause
