from typing import Optional

import numpy as np

from highway_env.road.road import Road
from highway_env.vehicle.kinematics import Vehicle


def create_random_vehicle_highway(
        v_type,
        road: Road,
        speed: float = None,
        lane_from: Optional[str] = None,
        lane_to: Optional[str] = None,
        lane_id: Optional[int] = None,
        spacing: float = 1,
        exclude_emergency_lane: bool = False,
) \
        -> "Vehicle":
    """
    Create a random vehicle on the road.

    The lane and /or speed are chosen randomly, while longitudinal position is chosen behind the last
    vehicle in the road with density based on the number of lanes.

    :param v_type: the type of the vehicle to create
    :param road: the road where the vehicle is driving
    :param speed: initial speed in [m/s]. If None, will be chosen randomly
    :param lane_from: start node of the lane to spawn in
    :param lane_to: end node of the lane to spawn in
    :param lane_id: id of the lane to spawn in
    :param spacing: ratio of spacing to the front vehicle, 1 being the default
    :param exclude_emergency_lane: whether to exclude the emergency lane from the possible lanes to spawn in
    :return: A vehicle with random position and/or speed
    """
    _from = lane_from or road.np_random.choice(list(road.network.graph.keys()))
    _to = lane_to or road.np_random.choice(list(road.network.graph[_from].keys()))
    if exclude_emergency_lane:
        _id = lane_id if lane_id is not None else road.np_random.choice(
            list(range(0, len(road.network.graph[_from][_to]) - 1)))
    else:
        _id = lane_id if lane_id is not None else road.np_random.choice(len(road.network.graph[_from][_to]))
    lane = road.network.get_lane((_from, _to, _id))
    if speed is None:
        if lane.speed_limit is not None:
            speed = road.np_random.uniform(0.8 * lane.speed_limit, 1.1 * lane.speed_limit)
        else:
            speed = road.np_random.uniform(Vehicle.DEFAULT_INITIAL_SPEEDS[0], Vehicle.DEFAULT_INITIAL_SPEEDS[1])
    default_spacing = 12 + 1.0 * speed
    offset = spacing * default_spacing
    x0 = np.max([lane.local_coordinates(v.position)[0] for v in road.vehicles]) if len(road.vehicles) else 12
    x0 += offset * road.np_random.uniform(0.7, 1.3)
    v = v_type(road, lane.position(x0, 0), lane.heading_at(x0), speed)
    return v
