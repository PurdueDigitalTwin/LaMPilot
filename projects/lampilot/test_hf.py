import argparse
import warnings
from argparse import Namespace
from multiprocessing import Pool

from tqdm import tqdm

import projects.lampilot.utils as U
from projects.lampilot.utils.run import process_item_hf
from projects.lampilot.dt.dbl import *
from projects.lampilot.dt.hf_agent import HumanFeedbackCGAgent
from projects.lampilot.dt.vehicle_dt import CtrlVDT
from projects.lampilot.evaluator import get_evaluator_class, DbLEvaluator

warnings.simplefilter("ignore")

parser = argparse.ArgumentParser()

# Evaluator Config
parser.add_argument('--no-window', action='store_true')
parser.add_argument('--wait-time', type=float, default=1e-3)

# Run Config
parser.add_argument('--config-root', type=str, default='projects/lampilot/configs/DbLv1')
parser.add_argument('--model-name', type=str, default='gpt-3.5-turbo')
parser.add_argument('--ckpt-dir', type=str, default='ckpt/human-fdbk')
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
    demo_dataset = DbLv1DemoDataset(
        config_root=args.config_root,
        shuffle=args.shuffle,
        seed=args.random_seed,
    )
    dataset = DbLv1Dataset(
        config_root=args.config_root,
        shuffle=args.shuffle,
        seed=args.random_seed,
    ) if not args.use_demo else None

    results = U.load_results(args.ckpt_dir)
    vehicle_dt = CtrlVDT()
    hf_agent = HumanFeedbackCGAgent(
        model_name=args.model_name,
        ckpt_dir=args.ckpt_dir,
        resume=args.resume,
    )
    for item in demo_dataset:
        command, sample, iid = item['command'], item['sample'], item['id']
        if iid in [r['iid'] for r in results]:
            print(f"Skip {iid} since it has been evaluated.")
            continue
        print(f"{iid} is being evaluated...")
        evaluator_class = get_evaluator_class(sample['eval']['type'])
        evaluator: DbLEvaluator = evaluator_class(
            config=sample,
            show_window=True,
            wait_time=1e-5,
            record_video=args.record_video,
            video_dir=f"{args.ckpt_dir}/videos/{iid}",
        )
        hf_agent.reset(command=command, context_info=evaluator.get_context_info())

        critiques = []
        success = False
        result = None
        while not success:
            evaluator_class = get_evaluator_class(sample['eval']['type'])
            evaluator: DbLEvaluator = evaluator_class(
                config=sample,
                show_window=True,
                wait_time=1e-5,
                record_video=args.record_video,
                video_dir=f"{args.ckpt_dir}/videos/{iid}",
            )
            vehicle_dt.reset(ego_vehicle=evaluator.env.unwrapped.vehicle)
            policy = hf_agent.step()
            vehicle_dt.execute(policy)
            while not evaluator.ended:
                evaluator.step(vehicle_dt)
            evaluator.close()
            result = U.create_result_dict(iid, evaluator, code=policy, command=command)
            success, commit, critique = evaluator.check_task_success()
            hf_agent.receive_feedback(success, critique, commit=commit)
            critiques.append(critique)

        U.dump_json(result, f"{args.ckpt_dir}/cache/{iid}.json", indent=4)
        results.append(result)

    args_list = [
        (item['command'], item['sample'], item['id'], args.ckpt_dir, args)
        for item in dataset[:args.test_size]
        if item['id'] not in [r['iid'] for r in results]  # skip evaluated items
    ]

    with Pool(args.num_process) as pool:
        results += list(
            tqdm(
                pool.starmap(process_item_hf, args_list),
                total=len(args_list),
                desc="Processing items",
            )
        )

    U.dump_json(results, f"{args.ckpt_dir}/results.json", indent=4)
    U.compute_final_results(results)
