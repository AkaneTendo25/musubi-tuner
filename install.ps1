# Musubi Tuner - LTX-2 Branch Installer for Windows (PowerShell)
# Usage: irm https://raw.githubusercontent.com/[user]/[repo]/ltx-2/install.ps1 | iex

$ErrorActionPreference = "Stop"

Write-Host "=== Musubi Tuner LTX-2 Installer ===" -ForegroundColor Cyan
Write-Host ""

# Check if running in existing repo or need to clone
$inRepo = Test-Path "pyproject.toml"
if (-not $inRepo) {
    Write-Host "[1/6] Cloning repository..." -ForegroundColor Yellow
    $repoUrl = Read-Host "Enter repository URL (e.g., https://github.com/user/musubi-tuner.git)"
    git clone -b ltx-2 $repoUrl musubi-tuner
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to clone repository" -ForegroundColor Red
        exit 1
    }
    Set-Location musubi-tuner
} else {
    Write-Host "[1/6] Running in existing repository" -ForegroundColor Green
}

# Check Python version
Write-Host "[2/6] Checking Python version..." -ForegroundColor Yellow
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "Python not found! Please install Python 3.10 or later" -ForegroundColor Red
    Write-Host "Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    exit 1
}

$pythonVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
$versionNum = [double]$pythonVersion
if ($versionNum -lt 3.10) {
    Write-Host "Python $pythonVersion found, but 3.10+ is required" -ForegroundColor Red
    exit 1
}
Write-Host "Python $pythonVersion found" -ForegroundColor Green

# Create virtual environment
Write-Host "[3/6] Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to create virtual environment" -ForegroundColor Red
    exit 1
}

# Activate virtual environment
Write-Host "[4/6] Activating virtual environment..." -ForegroundColor Yellow
& "venv\Scripts\Activate.ps1"

# Install PyTorch
Write-Host "[5/6] Installing PyTorch 2.8.0 with CUDA 12.6..." -ForegroundColor Yellow
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126 --force-reinstall --no-deps
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to install PyTorch" -ForegroundColor Red
    exit 1
}

# Install Flash Attention (Windows prebuilt wheel)
Write-Host "[5/6] Installing Flash Attention..." -ForegroundColor Yellow
pip install https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.4.10/flash_attn-2.8.2+cu128torch2.8-cp310-cp310-win_amd64.whl
if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: Flash Attention installation failed (optional)" -ForegroundColor Yellow
}

# Install Musubi Tuner
Write-Host "[6/6] Installing Musubi Tuner and dependencies..." -ForegroundColor Yellow
pip install -e .
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to install Musubi Tuner" -ForegroundColor Red
    exit 1
}

pip install ascii-magic matplotlib tensorboard prompt-toolkit
if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: Some dev dependencies failed to install" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "=== Installation Complete! ===" -ForegroundColor Green
Write-Host ""
Write-Host "To activate the environment in the future, run:" -ForegroundColor Cyan
Write-Host "  venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "To deactivate, run:" -ForegroundColor Cyan
Write-Host "  deactivate" -ForegroundColor White
Write-Host ""
