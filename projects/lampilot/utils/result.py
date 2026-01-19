import numpy as np


def create_result_dict(iid: str, evaluator, show=True, code: dict = None, command: str = "",
                       context_info: str = "") -> dict:
    ret = {
        'iid': iid,
        'overall_score': evaluator.score,
        'ttc_score': evaluator.score_ttc,
        'speed_variance_score': evaluator.score_speed_variance,
        'time_efficiency_score': evaluator.score_time_efficiency,
        'success': evaluator.success,
        'collision': evaluator.collision,
        'command': command,
        'context': context_info,
    }
    if code is not None:
        ret.update(code)

    if show:
        print(f"# {iid}")
        print(f"Overall score: {ret['overall_score']:.1f}")
        print(f"TTC score: {ret['ttc_score']:.1f}")
        print(f"SV score: {ret['speed_variance_score']:.1f}")
        print(f"TE score: {ret['time_efficiency_score']:.1f}")
        print(f"Success: {ret['success']}")
        print(f"Collision: {ret['collision']}\n")
    return ret


def compute_final_results(results, show=True) -> dict:
    ttc_score = np.mean([r['ttc_score'] for r in results if r['success']])
    sv_score = np.mean([r['speed_variance_score'] for r in results if r['success']])
    te_score = np.mean([r['time_efficiency_score'] for r in results if r['success']])
    overall_score = np.mean([r['overall_score'] for r in results if r['success']])
    success_rate = np.mean([r['success'] for r in results])
    collision_rate = np.mean([r['collision'] for r in results])
    driving_score = success_rate * overall_score - collision_rate * 500
    if show:
        print(f"========================================\n")
        print(f"Total number of episodes: {len(results)}")
        print(f"TTC score: {ttc_score:.1f}")
        print(f"SV score: {sv_score:.1f}")
        print(f"TE score: {te_score:.1f}")
        print(f"Overall score: {overall_score:.1f}")
        print(f"Success rate: {success_rate * 100:.1f}%")
        print(f"Collision rate: {collision_rate * 100:.1f}%")
        print(f"Driving score: {driving_score:.1f}")
        print(f"========================================\n")
    return {
        'ttc_score': ttc_score,
        'sv_score': sv_score,
        'te_score': te_score,
        'overall_score': overall_score,
        'success_rate': success_rate,
        'collision_rate': collision_rate,
        'driving_score': driving_score,
    }
