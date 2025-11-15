#!/usr/bin/env bash
#
# Convenience wrapper to start the FastAPI HTTP server.
#
# Usage:
#   ./api.sh
#   ./api.sh --host 127.0.0.1 --port 8080
#
# Note: Additional uvicorn options can be passed as arguments.
#

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Activate the virtual environment
source "$SCRIPT_DIR/.venv/bin/activate"

# Run the API server
# If no arguments provided, use defaults from config
if [ $# -eq 0 ]; then
    exec python3 -m src.api_server
else
    # Allow passing uvicorn options
    exec uvicorn src.api_server:app "$@"
fi
