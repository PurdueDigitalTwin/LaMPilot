from projects.lampilot.evaluator.base import DbLEvaluator
from projects.lampilot.evaluator.lane_change import LaneChangeEval
from projects.lampilot.evaluator.acc import ACCEvalbySpeed, ACCEvalbyDistance
from projects.lampilot.evaluator.overtake import OvertakeEval
from projects.lampilot.evaluator.pullover import PullOverEval
from projects.lampilot.evaluator.intersection import IntersectionEval

# Mapping of evaluator type names to classes (safer than using eval())
EVALUATOR_CLASSES = {
    'DbLEvaluator': DbLEvaluator,
    'LaneChangeEval': LaneChangeEval,
    'ACCEvalbySpeed': ACCEvalbySpeed,
    'ACCEvalbyDistance': ACCEvalbyDistance,
    'OvertakeEval': OvertakeEval,
    'PullOverEval': PullOverEval,
    'IntersectionEval': IntersectionEval,
    # Legacy alias for backward compatibility
    'AccEval': ACCEvalbySpeed,
}


def get_evaluator_class(evaluator_type: str):
    """
    Safely get an evaluator class by name.
    
    Args:
        evaluator_type: Name of the evaluator class
        
    Returns:
        The evaluator class
        
    Raises:
        ValueError: If the evaluator type is not found
    """
    if evaluator_type not in EVALUATOR_CLASSES:
        raise ValueError(
            f"Unknown evaluator type: {evaluator_type}. "
            f"Available types: {list(EVALUATOR_CLASSES.keys())}"
        )
    return EVALUATOR_CLASSES[evaluator_type]
