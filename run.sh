#!/usr/bin/env bash
#
# Convenience wrapper to run Python scripts with virtual environment activated.
#
# Usage:
#   ./run.sh scripts/run_once.py --prompt "Hello"
#   ./run.sh -m src.chat_cli
#

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the virtual environment
source "$SCRIPT_DIR/.venv/bin/activate"

# Run python3 with all arguments passed to this script
exec python3 "$@"
