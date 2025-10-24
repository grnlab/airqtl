#!/bin/bash

# Exit on any error
set -e

# Install the project in development mode
echo "Installing in development mode..."
pip3 install --break-system-packages --no-cache-dir -e /workspace/

# Verify PyTorch installation
echo "Verifying PyTorch installation..."
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

git config --global --add safe.directory /workspace
