#!/usr/bin/env bash
#
# Convenience wrapper to start the enhanced chat CLI with Rich UI.
#
# Usage:
#   ./chat-rich.sh
#   ./chat-rich.sh --system-prompt "You are a helpful coding assistant"
#   ./chat-rich.sh --max-tokens 256 --temperature 0.8
#

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the virtual environment
source "$SCRIPT_DIR/.venv/bin/activate"

# Run the enhanced chat CLI with any passed arguments
exec python3 -m src.chat_cli_rich "$@"
