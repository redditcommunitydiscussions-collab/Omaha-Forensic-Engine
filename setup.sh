#!/bin/bash

# Setup script for The Omaha Forensic Engine
# This script creates a virtual environment and installs dependencies

echo "🏗️  Setting up The Omaha Forensic Engine..."

# Check if Python 3.11 is available
if ! command -v python3.11 &> /dev/null; then
    echo "❌ Python 3.11 is not installed. Please install Python 3.11 first."
    exit 1
fi

# Create virtual environment
echo "📦 Creating virtual environment..."
python3.11 -m venv venv

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Create .env file from template if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file from template..."
    cp .env.template .env
    echo "⚠️  Please edit .env and add your GOOGLE_API_KEY"
else
    echo "✅ .env file already exists"
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "To activate the virtual environment, run:"
echo "  source venv/bin/activate"
echo ""
echo "To run the application, run:"
echo "  streamlit run app.py"
echo ""
