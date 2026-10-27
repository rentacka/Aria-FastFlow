@echo off
chcp 65001 > nul
title Aria-FastFlow Pure-Clean Japanese LoRA Trainer (RTX 3060)
echo ===============================================================================
echo   Studio 34: Pure-Clean Japanese LoRA Apex Trainer (RTX 3060 12GB)
echo   Master's Vision: Zero Financial Words x Google Compliance x 480 tok/s
echo ===============================================================================
echo.

set PYTHON_ENV=H:\DiffusionQwen_env\Scripts\python.exe

if not exist "%PYTHON_ENV%" (
    echo [!] H:\DiffusionQwen_env not found. Using default python...
    set PYTHON_ENV=python
)

cd /d "%~dp0"

echo [1/2] Checking clean Japanese dataset...
if not exist "dataset_cache\aria_home_training_set_ja_clean.jsonl" (
    echo [*] Building clean dataset...
    "%PYTHON_ENV%" build_clean_japanese_dataset.py
)

echo.
echo [2/2] Training Clean Japanese LoRA (Target: 1,000 steps / ~9-10 min)...
echo.
"%PYTHON_ENV%" train_clean_japanese_diffusion_gemma.py

echo.
echo Aria: Clean Japanese LoRA training completed successfully!
pause
