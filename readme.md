# LaMPilot: An Open Benchmark Dataset for Autonomous Driving with Language Model Programs

**Authors**: Yunsheng Ma, Can Cui, Xu Cao, Wenqian Ye, Peiran Liu, Juanwu Lu, Amr Abdelraouf, Rohit Gupta, Kyungtae Han, Aniket Bera, James M. Rehg, Ziran Wang

## Abstract

Autonomous driving (AD) has made significant strides in recent years. However, existing frameworks struggle to interpret and execute spontaneous user instructions, such as "overtake the car ahead." Large Language Models (LLMs) have demonstrated impressive reasoning capabilities showing potential to bridge this gap. In this paper, we present LaMPilot, a novel framework that integrates LLMs into AD systems, enabling them to follow user instructions by generating code that leverages established functional primitives. We also introduce LaMPilot-Bench, the first benchmark dataset specifically designed to quantitatively evaluate the efficacy of language model programs in AD. Adopting the LaMPilot framework, we conduct extensive experiments to assess the performance of off-the-shelf LLMs on LaMPilot-Bench. Our results demonstrate the potential of LLMs in handling diverse driving scenarios and following user instructions in driving.


## 🚀 Features

- **Natural Language to Code**: Convert high-level driving commands into executable Python code
- **Policy Repository**: Automatically stores and retrieves successful driving policies for reuse
- **Human-in-the-Loop Feedback**: Incorporates human feedback to iteratively improve generated policies
- **Multiple LLM Support**: Compatible with GPT-3.5, GPT-4, CodeLlama, Llama-2, and Code-Bison
- **Flexible Evaluation**: Supports various driving tasks including lane changes, overtaking, intersection navigation, and more

## 📋 Table of Contents

- [Abstract](#abstract)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [LaMPilot-Bench](#-lampilot-bench)
- [Usage Examples](#usage-examples)
- [Configuration](#configuration)
- [Citation](#citation)
- [License](#license)

## 🔧 Installation

### Prerequisites

- Python 3.8+
- OpenAI API key (or access to other supported LLM services)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/PurdueDigitalTwin/LaMPilot.git
cd LaMPilot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API key:
```bash
export OPENAI_API_KEY=your_api_key_here
```

## 🚦 Quick Start

### Basic Usage

Run a single task with a configuration file:

```bash
# Using the helper script (recommended)
./run_demo.sh --config projects/lampilot/configs/DbLv1/go_straight.json

# Or directly with Python
python projects/lampilot/demo.py --config projects/lampilot/configs/DbLv1/go_straight.json
```

Additional options for demo:
- `--model-name`: Specify LLM model (default: `gpt-3.5-turbo`)
- `--zero-shot`: Use zero-shot mode (default: few-shot)
- `--no-window`: Disable visualization window
- `--wait-time`: Wait time between simulation steps (default: 1e-3)

### Human Feedback Agent

Run the human feedback agent for iterative policy improvement:

```bash
# Using the helper script (recommended)
./run_test_hf.sh \
    --config-root projects/lampilot/configs/DbLv1 \
    --model-name gpt-3.5-turbo \
    --ckpt-dir ckpt/human-fdbk \
    --resume

# Or directly with Python
python projects/lampilot/test_hf.py \
    --config-root projects/lampilot/configs/DbLv1 \
    --model-name gpt-3.5-turbo \
    --ckpt-dir ckpt/human-fdbk \
    --resume
```

Additional options:
- `--test-size`: Number of test cases to evaluate (default: 98)
- `--use-demo`: Use demo dataset instead of full dataset
- `--num-process`: Number of parallel processes (default: 1)
- `--few-shot`: Enable few-shot learning
- `--record-video`: Record simulation videos
- `--shuffle`: Shuffle the dataset
- `--random_seed`: Random seed for reproducibility (default: 42)

### Zero-Shot and Few-Shot Code Generation

Test code generation without policy repository:

```bash
# Demo: Run few-shot evaluation on 5 random scenarios with GPT-5.2
./run_test_icl.sh \
    --config-root projects/lampilot/configs/DbLv1 \
    --model-name gpt-5.2 \
    --test-size 5 \
    --few-shot \
    --shuffle \
    --random_seed 123

# Using the helper script (recommended)
./run_test_icl.sh \
    --config-root projects/lampilot/configs/DbLv1 \
    --model-name gpt-4 \
    --few-shot  # Use --few-shot for few-shot, omit for zero-shot

# Or directly with Python
python projects/lampilot/test_icl.py \
    --config-root projects/lampilot/configs/DbLv1 \
    --model-name gpt-4 \
    --few-shot  # Use --few-shot for few-shot, omit for zero-shot
```

### Running Full Benchmark

The LaMPilot-Bench (DbLv1) contains **4,900 test cases** total. Run the complete benchmark evaluation:

```bash
# Zero-shot evaluation (full benchmark: 4,900 items)
python projects/lampilot/test_icl.py \
    --config-root projects/lampilot/configs/DbLv1 \
    --model-name gpt-3.5-turbo \
    --test-size 4900 \
    --num-process 4

# Few-shot evaluation (full benchmark: 4,900 items)
python projects/lampilot/test_icl.py \
    --config-root projects/lampilot/configs/DbLv1 \
    --model-name gpt-4 \
    --few-shot \
    --test-size 4900 \
    --num-process 4
```

**Note**: You can use a smaller `--test-size` value (e.g., 98, 500, 1000) for faster evaluation or testing purposes. The script will automatically skip already-evaluated items if you use the `--resume` flag or run with the same checkpoint directory.

**Note**: The helper scripts (`run_demo.sh`, `run_test_hf.sh`, `run_test_icl.sh`) automatically handle:
- Virtual environment activation (if present)
- PYTHONPATH configuration
- Proper module imports

## 🏗️ Architecture

LaMPilot consists of several key components:

### 1. Code Generation Agent (`cg_agent.py`)
- Converts natural language commands into Python code
- Supports multiple LLM backends (OpenAI)
- Handles zero-shot and few-shot learning modes

### 2. Policy Repository (`policy_repo.py`)
- Stores successful driving policies with semantic descriptions
- Uses vector database (ChromaDB) for efficient policy retrieval
- Automatically indexes policies for reuse in similar scenarios

### 3. Human Feedback Agent (`hf_agent.py`)
- Extends the code generation agent with feedback mechanisms
- Incorporates human critiques to refine generated policies
- Commits successful policies to the repository

### 4. Vehicle Digital Twin (`vehicle_dt.py`)
- Executes generated Python code in the simulation environment
- Provides a safe execution environment for LLM-generated code
- Implements control interfaces for vehicle manipulation

### 5. Evaluators (`evaluator/`)
- Task-specific evaluators for different driving scenarios
- Metrics: Time-to-Collision (TTC), speed variance, time efficiency
- Supports ACC (by speed and by distance), lane change, overtaking, intersection, and pullover tasks
- Evaluator types: `AccEval`, `ACCEvalbySpeed`, `ACCEvalbyDistance`, `LaneChangeEval`, `OvertakeEval`, `IntersectionEval`, `PullOverEval`

### 6. Benchmark Dataset (`dbl.py`)
- `DbLv1Dataset`: Loads and manages the LaMPilot-Bench (Drive by Language) dataset
- `DbLv1DemoDataset`: Subset of demo cases for quick testing
- Supports shuffling and random seed configuration
- Automatically loads configurations from `config_list.txt`

## 📊 LaMPilot-Bench

LaMPilot-Bench (also referred to as **DbLv1** - **Drive by Language version 1**, where **DbL** stands for **Drive by Language**) is the first benchmark dataset specifically designed to quantitatively evaluate the efficacy of language model programs in autonomous driving. The benchmark includes **32 diverse driving scenarios**, each with multiple samples and commands, resulting in **4,900 total test cases** for comprehensive evaluation.

### Task Categories

1. **Speed Control**
   - Absolute speed adjustments (increase/decrease to specific speeds)
   - Relative speed adjustments (increase/decrease by specific amounts)

2. **Following Distance**
   - Absolute distance adjustments (increase/decrease to specific distances)
   - Relative distance adjustments (increase/decrease by specific amounts)

3. **Lane Changes**
   - Left lane change
   - Right lane change

4. **Overtaking**
   - Left overtake
   - Right overtake

5. **Intersection Navigation**
   - Turn left
   - Turn right
   - Go straight

6. **Maneuvers**
   - Pull over

### Evaluation Metrics

- **Safety Score**: Based on Time-to-Collision (TTC)
- **Speed Variance Score**: Measures driving smoothness
- **Time Efficiency Score**: Evaluates task completion time
- **Overall Score**: Weighted combination of the above metrics

## 💡 Usage Examples

### Example 1: Simple Command Execution

```python
import json
from projects.lampilot.dt.cg_agent import CodeGenerationAgent
from projects.lampilot.dt.vehicle_dt import CtrlVDT
from projects.lampilot.evaluator import ACCEvalbySpeed

# Initialize agent
agent = CodeGenerationAgent(
    model_name="gpt-3.5-turbo",
    zero_shot=False  # Use few-shot by default
)
vehicle_dt = CtrlVDT()

# Load configuration
with open("projects/lampilot/configs/DbLv1/go_straight.json", 'r') as f:
    config = json.load(f)
sample = config['samples'][0]
command = config['commands'][0]

# Create evaluator
evaluator_type = sample.get('eval', {}).get('type', 'AccEval')
evaluator = eval(evaluator_type)(config=sample, show_window=True)

# Generate and execute policy
agent.reset(command=command, context_info=evaluator.get_context_info())
policy = agent.step()

vehicle_dt.reset(ego_vehicle=evaluator.env.unwrapped.vehicle)
vehicle_dt.execute(policy)

# Run simulation
while not evaluator.ended:
    evaluator.step(vehicle_dt)

evaluator.close()
print(f"Score: {evaluator.score:.1f}")
```

### Example 2: Using Policy Repository with Human Feedback

```python
from projects.lampilot.dt.hf_agent import HumanFeedbackCGAgent
from projects.lampilot.dt.vehicle_dt import CtrlVDT
from projects.lampilot.evaluator import OvertakeEval

# Initialize agent with policy repository
agent = HumanFeedbackCGAgent(
    model_name="gpt-4",
    ckpt_dir="ckpt/my_experiment",
    resume=True  # Load existing policies
)
vehicle_dt = CtrlVDT()

# Create evaluator
evaluator = OvertakeEval(config=sample, show_window=True)

# Generate policy (automatically retrieves similar policies)
agent.reset(
    command="Overtake the vehicle in front using the left lane",
    context_info=evaluator.get_context_info()
)
policy = agent.step()

# Execute policy
vehicle_dt.reset(ego_vehicle=evaluator.env.unwrapped.vehicle)
vehicle_dt.execute(policy)

# Run simulation
while not evaluator.ended:
    evaluator.step(vehicle_dt)

evaluator.close()

# Provide feedback for iterative improvement
success, commit, critique = evaluator.check_task_success()
agent.receive_feedback(success, critique, commit=commit)
```

### Example 3: Using the Benchmark Dataset

```python
from projects.lampilot.dt.dbl import DbLv1Dataset, DbLv1DemoDataset

# Load full dataset
dataset = DbLv1Dataset(
    config_root="projects/lampilot/configs/DbLv1",
    shuffle=True,
    seed=42
)

# Load demo dataset (subset for quick testing)
demo_dataset = DbLv1DemoDataset(
    config_root="projects/lampilot/configs/DbLv1",
    shuffle=False
)

# Iterate through dataset
for item in dataset:
    command = item['command']
    sample = item['sample']
    iid = item['id']  # Unique identifier
    # Process each item...
```

## ⚙️ Configuration

### Environment Configuration

LaMPilot supports multiple driving environments:

- `DTHighwayEnv`: Multi-lane highway driving (5 lanes with optional emergency lane)
- `DTIntersectionEnv`: Intersection navigation with cross traffic
- `RampMergeEnv`: Highway on-ramp merging

Environments are configured through the sample configuration in each task JSON file.

### LLM Configuration

Supported models (via OpenAI API):
- `gpt-3.5-turbo` (OpenAI)
- `gpt-4` (OpenAI)
- `gpt-4-1106-preview` (OpenAI)
- `gpt-5.2` (OpenAI)

The system uses the OpenAI API, so ensure your `OPENAI_API_KEY` environment variable is set.

### API Configuration

The system uses a set of predefined APIs for vehicle control:

- **Ego APIs**: `get_ego_vehicle()`, `get_target_speed()`, etc.
- **Control APIs**: `set_target_speed()`, `set_target_lane()`, `autopilot()`, etc.
- **Perception APIs**: `detect_front_vehicle_in()`, `get_left_lane()`, etc.
- **Route APIs**: `turn_left_at_next_intersection()`, etc.

See `projects/lampilot/prompts/apis.py` for the complete API reference.

## 📁 Project Structure

```
LaMPilot/
├── highway_env/          # Base driving environment (highway-env)
├── projects/
│   └── lampilot/
│       ├── configs/      # Benchmark configurations
│       │   └── DbLv1/    # LaMPilot-Bench configurations
│       │       ├── config_list.txt  # List of all config files
│       │       └── *.json           # Individual task configurations
│       ├── dt/           # Decision Transformer components
│       │   ├── cg_agent.py      # Code generation agent
│       │   ├── hf_agent.py      # Human feedback agent
│       │   ├── dbl.py           # Benchmark dataset loader
│       │   ├── policy_repo.py   # Policy repository
│       │   └── vehicle_dt.py    # Vehicle digital twin
│       ├── evaluator/    # Task-specific evaluators
│       │   ├── base.py          # Base evaluator class
│       │   ├── acc.py           # ACC evaluators
│       │   ├── lane_change.py   # Lane change evaluator
│       │   ├── overtake.py      # Overtaking evaluator
│       │   ├── intersection.py  # Intersection evaluator
│       │   └── pullover.py      # Pullover evaluator
│       ├── envs/         # Custom driving environments
│       │   ├── dt_highway.py    # Highway environment
│       │   ├── dt_intersection.py  # Intersection environment
│       │   └── ramp_merge_env.py   # Ramp merge environment
│       ├── primitives/   # Reusable driving primitives
│       ├── prompts/      # LLM prompts and templates
│       ├── utils/        # Utility functions
│       ├── demo.py       # Single task demo script
│       ├── test_hf.py    # Human feedback testing script
│       └── test_icl.py   # In-context learning testing script
├── ckpt/                 # Checkpoint directory for results (gitignored)
│                         # Contains evaluation results, policy repository, and cache files
├── run_demo.sh          # Helper script for demo.py
├── run_test_hf.sh       # Helper script for test_hf.py
├── run_test_icl.sh      # Helper script for test_icl.py
├── requirements.txt
└── README.md
```

## 🔬 Advanced Usage

### Custom Evaluator

Create a custom evaluator for new tasks:

```python
from projects.lampilot.evaluator.base import DbLEvaluator

class MyCustomEval(DbLEvaluator):
    def check_task_success(self):
        # Implement custom success criteria
        # Returns: (success: bool, commit: bool, critique: str)
        return self.custom_check()
    
    def get_context_info(self):
        # Provide context information for LLM
        return "Custom context information"
```

### Adding New Primitives

Add reusable driving primitives in `projects/lampilot/primitives/`:

```python
def my_custom_maneuver():
    """Description of the maneuver."""
    # Implementation using vehicle control APIs
    yield autopilot()
```

### Parallel Processing

For large-scale evaluations, use multiprocessing:

```bash
python projects/lampilot/test_icl.py \
    --config-root projects/lampilot/configs/DbLv1 \
    --model-name gpt-3.5-turbo \
    --test-size 98 \
    --num-process 4 \
    --no-window  # Disable visualization for parallel runs
```

### Video Recording

Record simulation videos for analysis:

```bash
python projects/lampilot/test_icl.py \
    --config-root projects/lampilot/configs/DbLv1 \
    --model-name gpt-3.5-turbo \
    --record-video \
    --ckpt-dir ckpt/my_experiment
```

Videos will be saved in `{ckpt_dir}/videos/{task_id}/`.

## 📝 Citation

If you use LaMPilot or LaMPilot-Bench in your research, please cite:

```bibtex
@inproceedings{ma2024lampilot,
  title={Lampilot: An open benchmark dataset for autonomous driving with language model programs},
  author={Ma, Yunsheng and Cui, Can and Cao, Xu and Ye, Wenqian and Liu, Peiran and Lu, Juanwu and Abdelraouf, Amr and Gupta, Rohit and Han, Kyungtae and Bera, Aniket and others},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={15141--15151},
  year={2024}
}
```

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 🙏 Acknowledgments

- Built on [highway-env](https://github.com/eleurent/highway-env) by Edouard Leurent
- Inspired by Voyager's skill repository architecture

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions and issues, please open an issue on GitHub.
