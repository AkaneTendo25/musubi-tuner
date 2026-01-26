#!/bin/bash
# Musubi Tuner - LTX-2 Branch Installer for Linux/macOS/WSL
# Usage: curl -fsSL https://raw.githubusercontent.com/[user]/[repo]/ltx-2/install.sh | bash

set -e

echo "=== Musubi Tuner LTX-2 Installer ==="
echo ""

# Check if running in existing repo or need to clone
if [ ! -f "pyproject.toml" ]; then
    echo "[1/6] Cloning repository..."
    read -p "Enter repository URL (e.g., https://github.com/user/musubi-tuner.git): " repo_url
    git clone -b ltx-2 "$repo_url" musubi-tuner
    cd musubi-tuner
else
    echo "[1/6] Running in existing repository"
fi

# Check Python version
echo "[2/6] Checking Python version..."
if ! command -v python3 &> /dev/null; then
    echo "Python3 not found! Please install Python 3.10 or later"
    exit 1
fi

python_version=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
required_version="3.10"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "Python $python_version found, but 3.10+ is required"
    exit 1
fi
echo "Python $python_version found"

# Create virtual environment
echo "[3/6] Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "[4/6] Activating virtual environment..."
source venv/bin/activate

# Detect CUDA version
echo "[5/6] Detecting CUDA version..."
cuda_version=""
if command -v nvcc &> /dev/null; then
    cuda_output=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p')
    cuda_major=$(echo $cuda_output | cut -d. -f1)
    cuda_minor=$(echo $cuda_output | cut -d. -f2)
    echo "CUDA $cuda_major.$cuda_minor detected"

    if [ "$cuda_major" -eq 12 ] && [ "$cuda_minor" -ge 6 ]; then
        cuda_version="cu126"
    elif [ "$cuda_major" -eq 12 ] && [ "$cuda_minor" -ge 4 ]; then
        cuda_version="cu124"
    else
        echo "Warning: CUDA version may not be optimal. Defaulting to cu126"
        cuda_version="cu126"
    fi
else
    echo "CUDA not detected. Defaulting to cu126"
    cuda_version="cu126"
fi

# Install PyTorch
echo "[5/6] Installing PyTorch 2.8.0 with CUDA $cuda_version..."
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/$cuda_version --force-reinstall --no-deps

# Install Flash Attention (build from source on Linux)
echo "[5/6] Installing Flash Attention..."
echo "Note: This may take several minutes to compile..."
pip install flash-attn --no-build-isolation || echo "Warning: Flash Attention installation failed (optional)"

# Install Musubi Tuner
echo "[6/6] Installing Musubi Tuner and dependencies..."
pip install -e .
pip install ascii-magic matplotlib tensorboard prompt-toolkit

echo ""
echo "=== Installation Complete! ==="
echo ""
echo "To activate the environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "  deactivate"
echo ""
