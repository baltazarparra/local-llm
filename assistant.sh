#!/usr/bin/env bash
#
# Launch the Personal Assistant with Memory
#
# Usage:
#   ./assistant.sh [options]
#
# Options are passed directly to the Python script.
# Examples:
#   ./assistant.sh
#   ./assistant.sh --db my_memory.db
#   ./assistant.sh --model Qwen/Qwen2.5-3B-Instruct
#

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Change to project directory
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
elif [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
else
    echo "Warning: No virtual environment found (.venv or venv)"
    echo "Consider creating one with: python3 -m venv .venv"
fi

# Check if required dependencies are installed
if ! python -c "import sentence_transformers" 2>/dev/null; then
    echo "Error: sentence-transformers not installed"
    echo "Please install dependencies: pip install -r requirements.txt"
    exit 1
fi

# Run the assistant
echo "Starting Personal Assistant..."
echo ""
python -m src.assistant_chat "$@"

