import numpy as np

from highway_env.envs import AbstractEnv
from highway_env.utils import not_zero
from highway_env.vehicle.kinematics import Vehicle


def iid_to_sample_id(iid: str) -> str:
    return iid.split('_c')[0]


def compute_ttc(env: AbstractEnv) -> float:
    unwrapped_env = env.unwrapped if hasattr(env, 'unwrapped') else env
    ego_vehicle: Vehicle = unwrapped_env.vehicle
    ego_speed = ego_vehicle.speed
    ttc = -1.

    for other in unwrapped_env.road.vehicles:
        if (other is ego_vehicle) or (ego_speed == other.speed):
            continue
        # margin = ego_vehicle.LENGTH / 2
        # distance = ego_vehicle.lane_distance_to(other) - margin
        distance = ego_vehicle.lane_distance_to(other)
        other_projected_speed = other.speed * np.dot(other.direction, ego_vehicle.direction)
        time_to_collision = distance / not_zero(ego_speed - other_projected_speed)
        if time_to_collision < 0:
            continue
        ttc = min(ttc, time_to_collision) if ttc > 0 else time_to_collision
    return ttc


def ordinal(n: int):
    if 10 <= n % 100 < 20:
        return str(n) + 'th'
    else:
        return str(n) + {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, "th")
