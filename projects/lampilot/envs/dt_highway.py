from highway_env.envs.merge_env import *
from highway_env.utils import class_from_path
from .utils import create_random_vehicle_highway


# noinspection PyUnresolvedReferences,PyTypeChecker
class DTHighwayEnv(MergeEnv):
    """
    a highway environment.

    The ego-vehicle starts on a highway.
    """

    def _reward(self, action: int) -> float:
        return 0.0

    def _rewards(self, action: int) -> Dict[Text, float]:
        return {'total_reward': 0.0}

    @classmethod
    def default_config(cls) -> dict:
        cfg = super().default_config()
        vehicles_count = 20
        cfg.update({
            "action": {
                "type": "ContinuousAction",
            },
            "stage_length": [2000],
            "num_lanes": 5,
            "emergency_lane": True,
            "screen_width": 2000,
            "screen_height": 300,
            "vehicles_count": vehicles_count,
            "vehicles_density": 1.0,
            "truncate_after_meter": 2400,
            "duration": 1000,
            "show_trajectories": True,
            "policy_frequency": 10,
            "simulation_frequency": 10,
            "ego_vehicle": {
                "speed": 20,
                "start_lane_index": ('a', 'b', 0),
                "start_position": [380, 0]
            }
        })
        return cfg

    def _is_terminated(self) -> bool:
        """The episode is over when a collision occurs or when the access ramp has been passed."""
        return self.vehicle.crashed or not self.vehicle.on_road

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached or if reached the end of the road."""
        return self.time >= self.config["duration"] or bool(
            self.vehicle.position[0] > self.config["truncate_after_meter"])

    def _make_vehicles(self) -> None:
        """
        Populate the road with several vehicles on the highway and on the merging lane, as well as an ego-vehicle.
        """

        ego_vehicle_type = self.action_type.vehicle_class

        ego_vehicle = ego_vehicle_type(
            self.road,
            self.road.network.get_lane(self.config["ego_vehicle"]["start_lane_index"]).position(
                self.config["ego_vehicle"]["start_position"][0],
                self.config["ego_vehicle"]["start_position"][1]),
            speed=self.config["ego_vehicle"]["speed"]
        )

        other_vehicles_type = class_from_path(self.config["other_vehicles_type"])

        for i in range(self.config["vehicles_count"]):
            vehicle = create_random_vehicle_highway(
                other_vehicles_type,
                self.road,
                spacing=1 / self.config["vehicles_density"],
                lane_from='a' if i == 0 else None,
                lane_to='b' if i == 0 else None,
                exclude_emergency_lane=self.config["emergency_lane"]
            )
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

    def _make_road(self) -> None:
        """
        Make a road.

        :return: the road
        """
        net = RoadNetwork()

        highway_speed_limit = 31.2928  # 70 mph
        ramp_speed_limit = 24.5872  # 55 mph

        highway = self.config["stage_length"]

        # Highway lanes
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        line_type = [(c, s)]
        if self.config["emergency_lane"]:
            line_type.extend([(n, s)] * (self.config["num_lanes"] - 2))
            line_type.extend([(n, c)] * 2)
        elif self.config["emergency_lane"] == False:
            line_type.extend([(n, s)] * (self.config["num_lanes"] - 1))
            line_type.extend([(n, c)])

        num_regular_lane = self.config["num_lanes"] - 1 if self.config["emergency_lane"] else self.config["num_lanes"]
        for i in range(num_regular_lane):
            net.add_lane("a", "b",
                         StraightLane([0, (i) * y[1]],
                                      [sum(highway), (i) * y[1]],
                                      line_types=line_type[i],
                                      speed_limit=highway_speed_limit)
                         )

        if self.config["emergency_lane"]:
            Emergency_Lane = StraightLane([0, num_regular_lane * y[1]],
                                          [sum(highway), num_regular_lane * y[1]],
                                          line_types=(c, c),
                                          forbidden=True,
                                          speed_limit=highway_speed_limit)
            net.add_lane("a", "b", Emergency_Lane)

        road = Road(network=net,
                    np_random=self.np_random,
                    record_history=self.config["show_trajectories"])

        self.road = road
