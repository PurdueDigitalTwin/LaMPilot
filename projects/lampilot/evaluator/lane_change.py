from .base import *
from highway_env.road.road import LaneIndex


class LaneChangeEval(DbLEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_lane_index: LaneIndex = self.ego_vehicle.lane_index
        _from, _to, _id = self.start_lane_index
        road = self.ego_vehicle.road
        if self.config['eval']['direction'] == 'left':
            direction = -1
        elif self.config['eval']['direction'] == 'right':
            direction = 1
        else:
            raise ValueError(f"Invalid direction {self.config['eval']['direction']}")

        self.target_lane_index: LaneIndex = (
            _from, _to, int(np.clip(_id + direction, 0, len(road.network.graph[_from][_to]) - 1)))
        self.target_lane = road.network.get_lane(self.target_lane_index)

        assert self.target_lane_index != self.start_lane_index, \
            f"Target lane index {self.target_lane_index} is the same as start lane index {self.start_lane_index}"
        assert road.network.get_lane(self.target_lane_index).is_reachable_from(self.ego_vehicle.position), \
            f"Target lane index {self.target_lane_index} is not reachable from current position {self.ego_vehicle.position}"

    def step(self, agent: VehicleDigitalTwin):
        super().step(agent)
        if self.ego_vehicle.lane_index == self.target_lane_index:
            ego_heading = self.ego_vehicle.heading
            lane_coords = self.target_lane.local_coordinates(self.ego_vehicle.position)
            lane_heading = self.target_lane.heading_at(lane_coords[0])
            if np.abs(ego_heading - lane_heading) < np.deg2rad(5):  # 5 degrees
                self.done = True
                self.success = True
