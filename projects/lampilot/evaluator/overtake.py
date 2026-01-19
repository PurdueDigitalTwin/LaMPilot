from .base import *
from highway_env.road.road import LaneIndex


class OvertakeEval(DbLEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config['eval']['direction'] == 'left':
            direction = -1
        elif self.config['eval']['direction'] == 'right':
            direction = 1
        else:
            raise ValueError(f"Invalid direction {self.config['eval']['direction']}")
        self.start_lane_index: LaneIndex = self.ego_vehicle.lane_index
        _from, _to, _id = self.start_lane_index
        road = self.ego_vehicle.road
        self.target_lane_index: LaneIndex = (
            _from, _to, int(np.clip(_id + direction, 0, len(road.network.graph[_from][_to]) - 1)))
        self.target_lane = road.network.get_lane(self.target_lane_index)
        self.front_vehicle = self.ego_vehicle.road.neighbour_vehicles(self.ego_vehicle, self.ego_vehicle.lane_index)[0]

        assert self.target_lane_index != self.start_lane_index, \
            f"Target lane index {self.target_lane_index} is the same as start lane index {self.start_lane_index}"
        assert road.network.get_lane(self.target_lane_index).is_reachable_from(self.ego_vehicle.position), \
            f"Target lane index {self.target_lane_index} is not reachable from current position {self.ego_vehicle.position}"

        assert self.front_vehicle != None, \
            f"No front vehicle detected."
        start_lane = road.network.get_lane(self.start_lane_index)
        assert np.abs(self.ego_vehicle.lane_distance_to(self.front_vehicle, start_lane)) < 100, \
            (f"Front vehicle is too far away from the ego vehicle, "
             f"the distance is {self.ego_vehicle.lane_distance_to(self.front_vehicle, start_lane)}")

    def step(self, agent: VehicleDigitalTwin):
        super().step(agent)
        if self.ego_vehicle.lane_index == self.target_lane_index \
                and self.ego_vehicle.lane_distance_to(self.front_vehicle,
                                                      self.target_lane) < -2 * self.ego_vehicle.LENGTH:
            self.done = True
            self.success = True
