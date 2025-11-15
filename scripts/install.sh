#!/usr/bin/env bash
#
# Optimized installation script for ego_proxy dependencies
# Automatically detects and uses uv if available, otherwise falls back to optimized pip
#
# Usage:
#   ./scripts/install.sh              # Install core dependencies
#   ./scripts/install.sh --dev        # Install core + dev dependencies
#   ./scripts/install.sh --help       # Show help

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REQUIREMENTS_FILE="$PROJECT_ROOT/requirements.txt"
REQUIREMENTS_DEV_FILE="$PROJECT_ROOT/requirements-dev.txt"

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
INSTALL_DEV=false
SHOW_HELP=false

for arg in "$@"; do
    case $arg in
        --dev)
            INSTALL_DEV=true
            shift
            ;;
        --help|-h)
            SHOW_HELP=true
            shift
            ;;
        *)
            echo "Unknown option: $arg"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

if [ "$SHOW_HELP" = true ]; then
    echo "ego_proxy Installation Script"
    echo ""
    echo "Usage:"
    echo "  ./scripts/install.sh              Install core dependencies"
    echo "  ./scripts/install.sh --dev         Install core + dev dependencies"
    echo ""
    echo "This script automatically:"
    echo "  - Detects and uses uv if available (fastest)"
    echo "  - Falls back to optimized pip with --prefer-binary flags"
    echo "  - Shows progress and estimated time"
    exit 0
fi

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo -e "${YELLOW}Warning: Virtual environment not detected.${NC}"
    echo "It's recommended to activate your virtual environment first:"
    echo "  source .venv/bin/activate"
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check if requirements file exists
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo -e "${YELLOW}Error: requirements.txt not found at $REQUIREMENTS_FILE${NC}"
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Detect installation method
USE_UV=false
if command_exists uv; then
    USE_UV=true
    echo -e "${GREEN}✓ Detected uv - using ultra-fast installation${NC}"
else
    echo -e "${BLUE}ℹ uv not found - using optimized pip${NC}"
    echo "  Tip: Install uv for 10-100x faster installs: curl -LsSf https://astral.sh/uv/install.sh | sh"
fi

echo ""
echo -e "${BLUE}Installing dependencies...${NC}"
if [ "$INSTALL_DEV" = true ]; then
    echo -e "${BLUE}  Mode: Core + Development dependencies${NC}"
else
    echo -e "${BLUE}  Mode: Core dependencies only${NC}"
fi
echo ""

# Record start time
START_TIME=$(date +%s)

# Install using uv
if [ "$USE_UV" = true ]; then
    if [ "$INSTALL_DEV" = true ]; then
        uv pip install -r "$REQUIREMENTS_FILE" -r "$REQUIREMENTS_DEV_FILE"
    else
        uv pip install -r "$REQUIREMENTS_FILE"
    fi
else
    # Install using optimized pip
    PIP_FLAGS="--prefer-binary --upgrade-strategy only-if-needed"
    
    # Use pip cache if available
    if [ -d "$HOME/.cache/pip" ]; then
        PIP_FLAGS="$PIP_FLAGS --cache-dir $HOME/.cache/pip"
    fi
    
    if [ "$INSTALL_DEV" = true ]; then
        pip install $PIP_FLAGS -r "$REQUIREMENTS_FILE" -r "$REQUIREMENTS_DEV_FILE"
    else
        pip install $PIP_FLAGS -r "$REQUIREMENTS_FILE"
    fi
fi

# Calculate elapsed time
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

echo ""
if [ $MINUTES -gt 0 ]; then
    echo -e "${GREEN}✓ Installation completed in ${MINUTES}m ${SECONDS}s${NC}"
else
    echo -e "${GREEN}✓ Installation completed in ${SECONDS}s${NC}"
fi

echo ""
echo -e "${GREEN}Setup complete! You can now run:${NC}"
echo "  ./chat-rich.sh"

