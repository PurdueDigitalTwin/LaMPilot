#!/bin/bash
# Pre-release validation script for LaMPilot
# This script runs basic checks to ensure the codebase is ready for release

set -e  # Exit on error

echo "=========================================="
echo "LaMPilot Pre-Release Validation"
echo "=========================================="

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Activate virtual environment if it exists
if [ -d "$SCRIPT_DIR/venv" ]; then
    source "$SCRIPT_DIR/venv/bin/activate"
    echo "✓ Virtual environment activated"
fi

# Set PYTHONPATH to include project root
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

echo ""
echo "1. Checking Python syntax..."
python -m py_compile projects/lampilot/demo.py
python -m py_compile projects/lampilot/test_icl.py
python -m py_compile projects/lampilot/test_hf.py
python -m py_compile projects/lampilot/test_idm.py
echo "✓ All Python files have valid syntax"

echo ""
echo "2. Checking module imports..."
python -c "
import sys
sys.path.insert(0, '.')
import highway_env
from projects.lampilot.dt.cg_agent import CodeGenerationAgent
from projects.lampilot.dt.hf_agent import HumanFeedbackCGAgent
from projects.lampilot.dt.vehicle_dt import CtrlVDT
from projects.lampilot.dt.dbl import DbLv1Dataset, DbLv1DemoDataset
from projects.lampilot.dt.policy_repo import PolicyRepository
from projects.lampilot.evaluator import *
print('✓ All modules imported successfully')
"

echo ""
echo "3. Validating configuration files..."
python -c "
import json
import os
config_root = 'projects/lampilot/configs/DbLv1'
config_list_path = os.path.join(config_root, 'config_list.txt')
if not os.path.exists(config_list_path):
    raise FileNotFoundError(f'Config list not found: {config_list_path}')

with open(config_list_path, 'r') as f:
    config_files = [line.strip() for line in f if line.strip()]

for config_file in config_files:
    config_path = os.path.join(config_root, config_file)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f'Config file not found: {config_path}')
    with open(config_path, 'r') as f:
        config = json.load(f)
    if 'samples' not in config or 'commands' not in config:
        raise ValueError(f'Invalid config structure in {config_file}')

print(f'✓ Validated {len(config_files)} configuration files')
"

echo ""
echo "4. Testing dataset loading..."
python -c "
import sys
sys.path.insert(0, '.')
from projects.lampilot.dt.dbl import DbLv1Dataset, DbLv1DemoDataset

# Test full dataset
dataset = DbLv1Dataset(
    config_root='projects/lampilot/configs/DbLv1',
    shuffle=False,
    seed=42
)
print(f'✓ Full dataset loaded: {len(dataset)} items')

# Test demo dataset
demo_dataset = DbLv1DemoDataset(
    config_root='projects/lampilot/configs/DbLv1',
    shuffle=False
)
print(f'✓ Demo dataset loaded: {len(demo_dataset)} items')
"

echo ""
echo "5. Checking helper scripts..."
for script in run_demo.sh run_test_hf.sh run_test_icl.sh; do
    if [ -f "$script" ] && [ -x "$script" ]; then
        echo "✓ $script is executable"
    else
        echo "⚠ Warning: $script not found or not executable"
    fi
done

echo ""
echo "6. Checking required dependencies..."
python -c "
import importlib
required_packages = [
    'tqdm', 'chromadb', 'langchain', 'langchain_community', 
    'langchain_openai', 'highway_env', 'moviepy', 'openai', 'tiktoken'
]
missing = []
for pkg in required_packages:
    try:
        importlib.import_module(pkg.replace('-', '_'))
    except ImportError:
        missing.append(pkg)

if missing:
    raise ImportError(f'Missing packages: {missing}')
print('✓ All required dependencies are installed')
"

echo ""
echo "=========================================="
echo "✓ All validation checks passed!"
echo "=========================================="
echo ""
echo "Note: This script performs basic validation only."
echo "For full testing, consider running:"
echo "  ./run_demo.sh --config projects/lampilot/configs/DbLv1/go_straight.json --no-window"
echo ""
