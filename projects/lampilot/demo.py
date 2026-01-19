import argparse
import json
import os
import sys
import warnings

# Add project root to Python path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import highway_env

from projects.lampilot.dt.cg_agent import CodeGenerationAgent
from projects.lampilot.dt.vehicle_dt import CtrlVDT
from projects.lampilot.evaluator import get_evaluator_class, DbLEvaluator

warnings.simplefilter("ignore")

parser = argparse.ArgumentParser(description="Demo script for running a single task with a configuration file")
parser.add_argument('--config', type=str, required=True, help='Path to configuration JSON file')
parser.add_argument('--model-name', type=str, default='gpt-3.5-turbo', help='Model name to use (default: gpt-3.5-turbo)')
parser.add_argument('--zero-shot', action='store_true', help='Use zero-shot mode (default: few-shot)')
parser.add_argument('--no-window', action='store_true', help='Disable visualization window')
parser.add_argument('--wait-time', type=float, default=1e-3, help='Wait time between steps')

args = parser.parse_args()

if __name__ == '__main__':
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Get first sample and command
    if 'samples' in config and len(config['samples']) > 0:
        sample = config['samples'][0]
    else:
        raise ValueError("Configuration file must contain at least one sample")
    
    if 'commands' in config and len(config['commands']) > 0:
        command = config['commands'][0]
    else:
        raise ValueError("Configuration file must contain at least one command")
    
    # Initialize agent
    agent = CodeGenerationAgent(
        model_name=args.model_name,
        zero_shot=args.zero_shot,
    )
    vehicle_dt = CtrlVDT()
    
    # Create evaluator
    evaluator_type = sample.get('eval', {}).get('type', 'ACCEvalbySpeed')
    evaluator_class = get_evaluator_class(evaluator_type)
    evaluator: DbLEvaluator = evaluator_class(
        config=sample,
        show_window=not args.no_window,
        wait_time=args.wait_time,
    )
    
    # Generate and execute policy
    print(f"Command: {command}")
    agent.reset(command=command, context_info=evaluator.get_context_info())
    policy = agent.step()
    
    vehicle_dt.reset(ego_vehicle=evaluator.env.unwrapped.vehicle)
    vehicle_dt.execute(policy)
    
    # Run simulation
    while not evaluator.ended:
        evaluator.step(vehicle_dt)
    
    evaluator.close()
    
    # Print results
    print(f"\nTask completed!")
    print(f"Score: {evaluator.score:.1f}")
    if hasattr(evaluator, 'check_task_success'):
        success, commit, critique = evaluator.check_task_success()
        print(f"Success: {success}")
        if critique:
            print(f"Critique: {critique}")
