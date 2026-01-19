import argparse
import os.path
import warnings
from argparse import Namespace
from multiprocessing import Pool
from tqdm import tqdm

# Import highway_env to register all environments
import highway_env

import projects.lampilot.utils as U
from projects.lampilot.utils.run import process_item
from projects.lampilot.dt.cg_agent import CodeGenerationAgent
from projects.lampilot.dt.dbl import *
from projects.lampilot.dt.vehicle_dt import CtrlVDT
from projects.lampilot.evaluator import *

warnings.simplefilter("ignore")

parser = argparse.ArgumentParser()

# Evaluator Config
parser.add_argument('--no-window', action='store_true')
parser.add_argument('--wait-time', type=float, default=1e-3)

# Run Config
parser.add_argument('--config-root', type=str, default='projects/lampilot/configs/DbLv1')
parser.add_argument('--model-name', type=str, default='gpt-3.5-turbo')
parser.add_argument('--ckpt-dir', type=str, default='ckpt/zero-shot')
parser.add_argument('--test-size', type=int, default=98)
parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--resume', action='store_true')
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--use-demo', action='store_true')
parser.add_argument('--num-process', type=int, default=1)
parser.add_argument('--few-shot', action='store_true')
parser.add_argument('--record-video', action='store_true')

args: Namespace = parser.parse_args()
args.ckpt_dir = f"{args.ckpt_dir}/{args.model_name}"

if __name__ == '__main__':
    dataset = DbLv1DemoDataset(
        config_root=args.config_root,
        shuffle=args.shuffle,
        seed=args.random_seed,
    ) if args.use_demo else DbLv1Dataset(
        config_root=args.config_root,
        shuffle=args.shuffle,
        seed=args.random_seed,
    )

    results = U.load_results(args.ckpt_dir)

    args_list = [
        (item['command'], item['sample'], item['id'], args.ckpt_dir, args)
        for item in dataset[:args.test_size]
        if item['id'] not in [r['iid'] for r in results]  # skip evaluated items
    ]

    with Pool(args.num_process) as pool:
        results += list(
            tqdm(
                pool.starmap(process_item, args_list),
                total=len(args_list),
                desc="Processing items",
            )
        )

    U.dump_json(results, f"{args.ckpt_dir}/results.json", indent=4)
    U.compute_final_results(results)
