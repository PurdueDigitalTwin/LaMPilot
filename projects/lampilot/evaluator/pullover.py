from .base import *
from highway_env.road.road import LaneIndex
from highway_env.envs.merge_env import *


class PullOverEval(DbLEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        _from, _to, _id = self.ego_vehicle.lane_index
        most_right_lane_index = (_from, _to, int(len(self.ego_vehicle.road.network.graph[_from][_to]) - 1))
        if (self.ego_vehicle.road.network.get_lane(most_right_lane_index).line_types ==
                (LineType.CONTINUOUS_LINE, LineType.CONTINUOUS_LINE)):
            self.emergency_lane_index = most_right_lane_index
        else:
            self.emergency_lane_index = None

        assert self.emergency_lane_index != None, \
            f"No emergency lane"

    def step(self, agent: VehicleDigitalTwin):
        super().step(agent)
        if self.ego_vehicle.lane_index == self.emergency_lane_index and np.isclose(self.ego_vehicle.speed, 0.0,
                                                                                   atol=5e-1):
            self.done = True
            self.success = True
