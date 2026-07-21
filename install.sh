#!/bin/bash
# OpenHammer LLM Studio Installation Script

set -e

echo "=============================================="
echo "  OpenHammer LLM Studio - Installation Script"
echo "=============================================="
echo ""

# Check Python version
echo "📌 Checking Python version..."
python3 --version || { echo "❌ Python 3 is required"; exit 1; }

# Create virtual environment (optional)
read -p "Create virtual environment? (y/n): " create_venv
if [ "$create_venv" = "y" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "✓ Virtual environment created and activated"
fi

# Install dependencies from requirements.txt
echo ""
echo "📦 Installing dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# Ask about PyTorch with CUDA support
echo ""
read -p "Install PyTorch with CUDA 12.4 support? (y/n): " install_torch
if [ "$install_torch" = "y" ]; then
    echo "📦 Installing PyTorch with CUDA 12.4..."
    pip install torch --index-url https://download.pytorch.org/whl/cu124
fi

# Create directories
echo ""
echo "📁 Creating directories..."
mkdir -p datasets config models experiments

# Set permissions
chmod +x llm_studio_cli.py 2>/dev/null || true

echo ""
echo "=============================================="
echo "  ✅ Installation Complete!"
echo "=============================================="
echo ""
echo "To run the application:"
echo ""
if [ "$create_venv" = "y" ]; then
    echo "  source venv/bin/activate"
fi
echo "  # Desktop GUI (if PySide6 installed):"
echo "  python ui/app.py"
echo ""
echo "  # Command Line Interface:"
echo "  python llm_studio_cli.py"
echo ""
echo "=============================================="
