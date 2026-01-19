from highway_env.envs.merge_env import *
from highway_env.utils import class_from_path
from .utils import create_random_vehicle_highway


# noinspection PyUnresolvedReferences,PyTypeChecker
class RampMergeEnv(MergeEnv):
    """
    a highway merge environment.

    The ego-vehicle starts on a ramp merging onto a highway.
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
            "stage_length": [500, 80, 80, 2000],  # before, converging, merge, after
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
                "start_lane_index": ('j', 'k', 0),
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
            )
            vehicle.randomize_behavior()
            self.road.vehicles.append(vehicle)

        self.road.vehicles.append(ego_vehicle)
        self.vehicle = ego_vehicle

    def _make_road(self) -> None:
        """
        Make a road composed of a straight highway and a merging lane.

        :return: the road
        """
        net = RoadNetwork()

        highway_speed_limit = 31.2928  # 70 mph
        ramp_speed_limit = 24.5872  # 55 mph

        before, converging, merge, after = self.config["stage_length"]

        # Highway lanes
        ends = [before, converging, merge, after]
        c, s, n = LineType.CONTINUOUS_LINE, LineType.STRIPED, LineType.NONE
        y = [0, StraightLane.DEFAULT_WIDTH]
        line_type = [(c, s), (n, c)]
        line_type_merge = [(c, s), (n, s)]
        for i in range(2):
            net.add_lane("a", "b",
                         StraightLane([0, y[i]],
                                      [sum(ends[:2]), y[i]],
                                      line_types=line_type[i],
                                      speed_limit=highway_speed_limit)
                         )
            net.add_lane("b", "c",
                         StraightLane([sum(ends[:2]), y[i]],
                                      [sum(ends[:3]), y[i]],
                                      line_types=line_type_merge[i],
                                      speed_limit=highway_speed_limit)
                         )
            net.add_lane("c", "d", StraightLane([sum(ends[:3]), y[i]],
                                                [sum(ends), y[i]],
                                                line_types=line_type[i],
                                                speed_limit=highway_speed_limit)
                         )

        # Ramp
        amplitude = 3.25
        ljk = StraightLane([0, 6.5 + 4 + 4],
                           [ends[0], 6.5 + 4 + 4],
                           line_types=(c, c),
                           forbidden=True,
                           speed_limit=ramp_speed_limit
                           )
        lkb = SineLane(ljk.position(ends[0], -amplitude),
                       ljk.position(sum(ends[:2]), -amplitude),
                       amplitude,
                       2 * np.pi / (2 * ends[1]),
                       np.pi / 2,
                       line_types=[c, c],
                       forbidden=True,
                       speed_limit=ramp_speed_limit)
        lbc = StraightLane(lkb.position(ends[1], 0),
                           lkb.position(ends[1], 0) + [ends[2], 0],
                           line_types=(n, c),
                           forbidden=True,
                           speed_limit=highway_speed_limit)
        net.add_lane("j", "k", ljk)
        net.add_lane("k", "b", lkb)
        net.add_lane("b", "c", lbc)
        road = Road(network=net,
                    np_random=self.np_random,
                    record_history=self.config["show_trajectories"])
        road.objects.append(Obstacle(road, lbc.position(ends[2], 0)))
        self.road = road
