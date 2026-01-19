from highway_env.road.road import LaneIndex
from .base import *


class ACCEvalbySpeed(DbLEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if 'speed' in self.config['eval']:
            self.desired_speed = self.config['eval']['speed']
        elif 'rel_speed' in self.config['eval']:
            self.desired_speed = self.ego_vehicle.speed + self.config['eval']['rel_speed']
        # The flag to indicate whether the ego vehicle is reaching the desired speed.
        self.last_time = -1
        # The time duration to judge weather the ego vehicle is successfully achieving ACC.
        self.time_duration = 5
        # The time limit to judge weather the ego vehicle is successfully achieving ACC.
        self.failure_time = 60
        # The maximum gap between the ego vehicle and
        # the front vehicle when the ego vehicle is following the front vehicle.
        self.max_gap = 100
        self.failure_start_time = self.overall_time

    def step(self, agent: VehicleDigitalTwin):
        super().step(agent)
        self.front_vehicle = self.ego_vehicle.road.neighbour_vehicles(self.ego_vehicle, self.ego_vehicle.lane_index)[0]
        if self.front_vehicle and self.ego_vehicle.lane_distance_to(self.front_vehicle) > 100:
            self.front_vehicle = None
        # The ego vehicle will be assumed to reach the desired speed from the initial speed.
        # if the ego vehicle is in the range of the desired speed,
        # and the distance to the front vehicle is closer than max_gap if there is a front vehicle.
        if self.last_time == -1 and self.front_vehicle is None and np.abs(
                self.ego_vehicle.speed - self.desired_speed) < 1:
            self.last_time = self.overall_time
        elif self.last_time == -1 and self.front_vehicle is not None and \
                self.front_vehicle.speed > self.desired_speed and np.abs(
            self.desired_speed - self.ego_vehicle.speed) < 1:
            self.last_time = self.overall_time
        elif self.last_time == -1 and self.front_vehicle is not None and self.front_vehicle.speed < self.desired_speed and \
                np.abs(self.front_vehicle.speed - self.ego_vehicle.speed) < 1 and \
                np.abs(self.ego_vehicle.lane_distance_to(self.front_vehicle)) < self.max_gap:
            self.last_time = self.overall_time
        # The ego vehicle keeps the desired speed.
        elif self.last_time > 0:
            if self.overall_time - self.last_time > self.time_duration:
                self.done = True
                self.success = True
            elif self.front_vehicle is None and np.abs(self.ego_vehicle.speed - self.desired_speed) > 1:
                self.last_time = -1
            elif (self.front_vehicle is not None and self.front_vehicle.speed > self.desired_speed) \
                    and np.abs(self.desired_speed - self.ego_vehicle.speed) > 1:
                self.last_time = -1
            elif self.front_vehicle is not None and self.front_vehicle.speed < self.desired_speed and \
                    np.abs(self.front_vehicle.speed - self.ego_vehicle.speed) > 1:
                self.last_time = -1

        # Out of the time limit.
        if self.overall_time - self.failure_start_time > self.failure_time:
            self.done = True
            self.success = False


class ACCEvalbyDistance(DbLEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.front_vehicle = self.ego_vehicle.road.neighbour_vehicles(self.ego_vehicle, self.ego_vehicle.lane_index)[0]
        if 'distance' in self.config['eval']:
            self.desired_distance = self.config['eval']['distance']
        if 'rel_distance' in self.config['eval']:
            self.desired_distance = self.config['eval']['rel_distance'] + \
                                    self.ego_vehicle.lane_distance_to(self.front_vehicle)
        # The flag to indicate whether the ego vehicle is reaching the desired distance.
        self.last_time = -1
        # The time duration to judge weather the ego vehicle is successfully achieving ACC.
        self.time_duration = 5
        # The time limit to judge weather the ego vehicle is successfully achieving ACC.
        self.failure_time = 60
        # The maximum gap between the ego vehicle and the front vehicle when the ego vehicle is following the front vehicle.
        self.max_gap = 60
        self.failure_start_time = self.overall_time

        self.init_ego_speed = self.ego_vehicle.speed
        self.dis_tol = 0.3 * self.desired_distance

        assert self.front_vehicle != None, \
            f"No front vehicle detected."

        self.start_lane_index: LaneIndex = self.ego_vehicle.lane_index
        road = self.ego_vehicle.road
        start_lane = road.network.get_lane(self.start_lane_index)
        assert np.abs(self.ego_vehicle.lane_distance_to(self.front_vehicle, start_lane)) < 100, \
            (f"Front vehicle is too far away from the ego vehicle, "
             f"the distance is {self.ego_vehicle.lane_distance_to(self.front_vehicle, start_lane)}")

    def step(self, agent: VehicleDigitalTwin):
        super().step(agent)
        if self.front_vehicle.lane_index != self.ego_vehicle.lane_index:
            self.front_vehicle = (
                self.ego_vehicle.road.neighbour_vehicles(self.ego_vehicle, self.ego_vehicle.lane_index))[0]

        #  There is no front vehicle, and the ego vehicle is not in the range of desired distance.
        if self.front_vehicle is None and self.last_time == -1:
            if np.isclose(self.ego_vehicle.speed, self.init_ego_speed, atol=0.1):
                self.last_time = self.overall_time
            else:
                self.last_time = -1
        # The ego vehicle is following a front vehicle, and the ego vehicle is not in the range of desired distance.
        elif self.front_vehicle is not None and self.last_time == -1:
            if ((self.desired_distance - self.dis_tol) < \
                    np.abs(self.ego_vehicle.lane_distance_to(self.front_vehicle)) < \
                    (self.desired_distance + self.dis_tol)):
                self.last_time = self.overall_time
            else:
                self.last_time = -1
        # The ego vehicle is in the desired distance.
        elif self.last_time > 0:
            if self.overall_time - self.last_time > self.time_duration:
                self.done = True
                self.success = True
            # The ego vehicle is not in the desired distance when the ego vehicle is following a front vehicle.
            elif self.front_vehicle is not None and \
                    (np.abs(self.ego_vehicle.lane_distance_to(self.front_vehicle)) <
                     (self.desired_distance - self.dis_tol)) \
                    or (np.abs(self.ego_vehicle.lane_distance_to(self.front_vehicle)) >
                        (self.desired_distance + self.dis_tol)):
                self.last_time = -1
            # The speed of ego vehicle changed when the ego vehicle is not following a front vehicle.
            elif self.front_vehicle is None and np.isclose(self.ego_vehicle.speed, self.init_ego_speed, atol=0.1):
                self.last_time = -1

        if self.overall_time - self.failure_start_time > self.failure_time:
            self.done = True
            self.success = False
