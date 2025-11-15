#!/usr/bin/env bash
#
# Convenience wrapper to start the interactive chat CLI.
#
# Usage:
#   ./chat.sh
#   ./chat.sh --system-prompt "You are a helpful assistant"
#   ./chat.sh --max-tokens 256 --temperature 0.8
#

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the virtual environment
source "$SCRIPT_DIR/.venv/bin/activate"

# Run the chat CLI with any passed arguments
exec python3 -m src.chat_cli "$@"
