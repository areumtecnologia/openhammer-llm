# OpenHammer LLM Studio Installation Script for Windows
# Run this script in PowerShell

Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  OpenHammer LLM Studio - Installation Script" -ForegroundColor Cyan
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""

# Check Python version
Write-Host "📌 Checking Python version..." -ForegroundColor Yellow
try {
    $pythonVersion = python --version 2>&1
    Write-Host "✓ $pythonVersion" -ForegroundColor Green
} catch {
    Write-Host "❌ Python 3 is required. Please install Python first." -ForegroundColor Red
    exit 1
}

# Create virtual environment (optional)
$createVenv = Read-Host "Create virtual environment? (y/n)"
if ($createVenv -eq "y") {
    Write-Host "📦 Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
    Write-Host "✓ Virtual environment created" -ForegroundColor Green
    Write-Host "  To activate later: .\venv\Scripts\Activate.ps1" -ForegroundColor Gray
}

# Install core dependencies
Write-Host ""
Write-Host "📦 Installing core dependencies..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Always install psutil for hardware detection
Write-Host "Installing psutil..." -ForegroundColor Gray
python -m pip install psutil

# Ask about GUI
Write-Host ""
$installGui = Read-Host "Install PySide6 for GUI? (y/n)"
if ($installGui -eq "y") {
    Write-Host "📦 Installing PySide6..." -ForegroundColor Yellow
    python -m pip install PySide6
}

# Ask about Hugging Face datasets
Write-Host ""
$installDatasets = Read-Host "Install Hugging Face datasets library? (y/n)"
if ($installDatasets -eq "y") {
    Write-Host "📦 Installing datasets..." -ForegroundColor Yellow
    python -m pip install datasets
}

# Create directories
Write-Host ""
Write-Host "📁 Creating directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "datasets" | Out-Null
New-Item -ItemType Directory -Force -Path "config" | Out-Null
New-Item -ItemType Directory -Force -Path "models" | Out-Null
New-Item -ItemType Directory -Force -Path "experiments" | Out-Null
Write-Host "✓ Directories created: datasets, config, models, experiments" -ForegroundColor Green

Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host "  ✅ Installation Complete!" -ForegroundColor Green
Write-Host "==============================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To run the application:" -ForegroundColor White
Write-Host ""
if ($createVenv -eq "y") {
    Write-Host "  # Activate virtual environment:" -ForegroundColor Yellow
    Write-Host "  .\venv\Scripts\Activate.ps1" -ForegroundColor Gray
    Write-Host ""
}
Write-Host "  # Desktop GUI (if PySide6 installed):" -ForegroundColor Yellow
Write-Host "  python ui\app.py" -ForegroundColor Gray
Write-Host ""
Write-Host "  # Command Line Interface:" -ForegroundColor Yellow
Write-Host "  python llm_studio_cli.py" -ForegroundColor Gray
Write-Host ""
Write-Host "==============================================" -ForegroundColor Cyan
