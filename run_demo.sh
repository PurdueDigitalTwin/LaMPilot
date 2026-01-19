#!/bin/bash
# Helper script to run demo.py with proper PYTHONPATH

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Activate virtual environment if it exists
if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi

# Set PYTHONPATH to include project root
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

# Run the script with all passed arguments
python "$SCRIPT_DIR/projects/lampilot/demo.py" "$@"
