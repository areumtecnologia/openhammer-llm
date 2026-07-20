#!/bin/bash
# LLM Studio Installation Script

set -e

echo "=============================================="
echo "  LLM Studio - Installation Script"
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

# Install core dependencies
echo ""
echo "📦 Installing core dependencies..."
pip install --upgrade pip

# Always install psutil for hardware detection
pip install psutil

# Ask about GUI
echo ""
read -p "Install PySide6 for GUI? (y/n): " install_gui
if [ "$install_gui" = "y" ]; then
    echo "📦 Installing PySide6..."
    pip install PySide6
fi

# Ask about Hugging Face datasets
echo ""
read -p "Install Hugging Face datasets library? (y/n): " install_datasets
if [ "$install_datasets" = "y" ]; then
    echo "📦 Installing datasets..."
    pip install datasets
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
