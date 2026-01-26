@echo off
REM Musubi Tuner - LTX-2 Branch Installer for Windows (CMD)
REM Usage: curl -fsSL https://raw.githubusercontent.com/[user]/[repo]/ltx-2/install.cmd -o install.cmd && install.cmd && del install.cmd

setlocal EnableDelayedExpansion

echo === Musubi Tuner LTX-2 Installer ===
echo.

REM Check if running in existing repo or need to clone
if not exist "pyproject.toml" (
    echo [1/6] Cloning repository...
    set /p repo_url="Enter repository URL (e.g., https://github.com/user/musubi-tuner.git): "
    git clone -b ltx-2 !repo_url! musubi-tuner
    if errorlevel 1 (
        echo Failed to clone repository
        exit /b 1
    )
    cd musubi-tuner
) else (
    echo [1/6] Running in existing repository
)

REM Check Python version
echo [2/6] Checking Python version...
python --version >nul 2>&1
if errorlevel 1 (
    echo Python not found! Please install Python 3.10 or later
    echo Download from: https://www.python.org/downloads/
    exit /b 1
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set python_version=%%i
echo Python %python_version% found

REM Create virtual environment
echo [3/6] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo Failed to create virtual environment
    exit /b 1
)

REM Activate virtual environment
echo [4/6] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install PyTorch
echo [5/6] Installing PyTorch 2.8.0 with CUDA 12.6...
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126 --force-reinstall --no-deps
if errorlevel 1 (
    echo Failed to install PyTorch
    exit /b 1
)

REM Install Flash Attention (Windows prebuilt wheel)
echo [5/6] Installing Flash Attention...
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.10/flash_attn-2.8.2+cu128torch2.8-cp310-cp310-win_amd64.whl
if errorlevel 1 (
    echo Warning: Flash Attention installation failed ^(optional^)
)

REM Install Musubi Tuner
echo [6/6] Installing Musubi Tuner and dependencies...
pip install -e .
if errorlevel 1 (
    echo Failed to install Musubi Tuner
    exit /b 1
)

pip install ascii-magic matplotlib tensorboard prompt-toolkit
if errorlevel 1 (
    echo Warning: Some dev dependencies failed to install
)

echo.
echo === Installation Complete! ===
echo.
echo To activate the environment in the future, run:
echo   venv\Scripts\activate.bat
echo.
echo To deactivate, run:
echo   deactivate
echo.
pause
