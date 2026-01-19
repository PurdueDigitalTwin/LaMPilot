from argparse import Namespace

# Import highway_env to register all environments (important for multiprocessing)
import highway_env

from projects.lampilot.dt.cg_agent import CodeGenerationAgent
from projects.lampilot.dt.hf_agent import HumanFeedbackCGAgent
from projects.lampilot.dt.vehicle_dt import CtrlVDT
from projects.lampilot.evaluator import get_evaluator_class, DbLEvaluator
from .io import dump_json
from .result import create_result_dict


def process_item(command: str, sample: dict, iid: str, output_dir: str, args: Namespace,
                 agent: CodeGenerationAgent = None):
    print(f"{iid} is being evaluated...")
    cache_path = f"{output_dir}/cache/{iid}.json"
    evaluator_class = get_evaluator_class(sample['eval']['type'])
    evaluator: DbLEvaluator = evaluator_class(
        config=sample,
        show_window=False if args.no_window or args.num_process > 1 else True,
        wait_time=1e-5,
        record_video=args.record_video,
        video_dir=f"{output_dir}/videos/{iid}",
    )
    if agent is None:
        agent = CodeGenerationAgent(
            model_name=args.model_name,
            zero_shot=not args.few_shot,
        )
    vehicle_dt = CtrlVDT()
    context_info = evaluator.get_context_info()
    agent.reset(
        command=command,
        context_info=context_info,
    )
    vehicle_dt.reset(
        ego_vehicle=evaluator.env.unwrapped.vehicle
    )
    policy = agent.step()
    vehicle_dt.execute(policy)
    while not evaluator.ended:
        evaluator.step(vehicle_dt)
    evaluator.close()
    result = create_result_dict(iid, evaluator, code=policy, command=command, context_info=context_info)
    dump_json(result, cache_path, indent=4)
    return result


def process_item_hf(command: str, sample: dict, iid: str, output_dir: str, args: Namespace,
                    agent: CodeGenerationAgent = None):
    print(f"{iid} is being evaluated...")
    cache_path = f"{output_dir}/cache/{iid}.json"
    evaluator_class = get_evaluator_class(sample['eval']['type'])
    evaluator: DbLEvaluator = evaluator_class(
        config=sample,
        show_window=False if args.no_window or args.num_process > 1 else True,
        wait_time=1e-5,
        record_video=args.record_video,
        video_dir=f"{output_dir}/videos/{iid}",
    )
    if agent is None:
        agent = HumanFeedbackCGAgent(
            model_name=args.model_name,
            ckpt_dir=args.ckpt_dir,
            resume=True,
        )
    vehicle_dt = CtrlVDT()
    context_info = evaluator.get_context_info()
    agent.reset(
        command=command,
        context_info=context_info,
    )
    vehicle_dt.reset(
        ego_vehicle=evaluator.env.unwrapped.vehicle
    )
    policy = agent.step()
    vehicle_dt.execute(policy)
    while not evaluator.ended:
        evaluator.step(vehicle_dt)
    evaluator.close()
    result = create_result_dict(iid, evaluator, code=policy, command=command, context_info=context_info)
    dump_json(result, cache_path, indent=4)
    return result
