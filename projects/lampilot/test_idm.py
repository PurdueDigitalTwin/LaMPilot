import argparse
import os.path
import warnings
from argparse import Namespace
from multiprocessing import Pool

from tqdm import tqdm

import projects.lampilot.utils as U
from projects.lampilot.dt.dbl import DbLv1Dataset
from projects.lampilot.dt.vehicle_dt import MOBILDT, IDMDT
from projects.lampilot.evaluator import get_evaluator_class, DbLEvaluator

warnings.simplefilter("ignore")

parser = argparse.ArgumentParser()

# Evaluator Config
parser.add_argument('--no-window', action='store_true')
parser.add_argument('--wait-time', type=float, default=1e-5)

# Run Config
parser.add_argument('--config-root', type=str, default='projects/lampilot/configs/DbLv1')
parser.add_argument('--ckpt-dir', type=str, default='ckpt/heuristic')
parser.add_argument('--test-size', type=int, default=10000)
parser.add_argument('--shuffle', action='store_true')
parser.add_argument('--random_seed', type=int, default=42)
parser.add_argument('--method', type=str, default='mobil')
parser.add_argument('--num-process', type=int, default=1)

args: Namespace = parser.parse_args()


def process_item(command, sample, iid, cache_path):
    global args
    print(f"{iid} is being evaluated...")
    evaluator_class = get_evaluator_class(sample['eval']['type'])
    evaluator: DbLEvaluator = evaluator_class(
        config=sample,
        show_window=False if args.no_window or args.num_process > 1 else True,
        wait_time=1e-5,
    )
    vehicle_dt = MOBILDT() if args.method == 'mobil' else IDMDT()
    vehicle_dt.reset(
        ego_vehicle=evaluator.env.unwrapped.vehicle
    )
    while not evaluator.ended:
        evaluator.step(vehicle_dt)
    evaluator.close()
    result = U.create_result_dict(iid, evaluator)
    U.dump_json(result, cache_path, indent=4)
    return result


if __name__ == '__main__':
    dataset = DbLv1Dataset(
        config_root=args.config_root,
        shuffle=args.shuffle,
        seed=args.random_seed,
    )
    print(dataset._command_stat())
    # print(f"Total number of data items: {len(dataset)}")

    results = []
    args.ckpt_dir = f"{args.ckpt_dir}/{args.method}"
    if os.path.exists(f"{args.ckpt_dir}/cache"):
        for file in os.listdir(f"{args.ckpt_dir}/cache"):
            if file.endswith(".json"):
                results.append(U.load_json(f"{args.ckpt_dir}/cache/{file}"))
        U.dump_json(results, f"{args.ckpt_dir}/results.json", indent=4)
    else:
        os.makedirs(f"{args.ckpt_dir}/cache", exist_ok=True)

    args_list = [
        (item['command'], item['sample'], item['id'], f"{args.ckpt_dir}/cache/{item['id']}.json")
        for item in dataset[:args.test_size]
        if U.iid_to_sample_id(item['id']) not in [U.iid_to_sample_id(r['iid']) for r in results]
    ]

    with Pool(args.num_process) as pool:
        results += list(
            tqdm(
                pool.starmap(process_item, args_list),
                total=len(args_list)
            )
        )

    U.dump_json(results, f"{args.ckpt_dir}/results.json", indent=4)
    U.compute_final_results(results)
