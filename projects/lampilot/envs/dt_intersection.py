from highway_env.vehicle.kinematics import Vehicle
from highway_env.road.lane import LineType, StraightLane, CircularLane, AbstractLane
from highway_env.road.regulation import RegulatedRoad
import numpy as np
from typing import Dict, Tuple, Text
from highway_env.envs.merge_env import *
from highway_env.utils import class_from_path
from .utils import create_random_vehicle_highway
from projects.lampilot.vehicle.objects import StopSign


class DTIntersectionEnv(AbstractEnv):

    @classmethod
    def default_config(cls) -> dict:
        config = super().default_config()
        config.update({
            "observation": {
                "type": "Kinematics",
                "vehicles_count": 10,
                "vehicles_density": 2.0,
                "features": ["presence", "x", "y", "vx", "vy", "long_off", "lat_off", "ang_off"],
                "features_range": {
                    "x": [-100, 100],
                    "y": [-100, 100],
                    "vx": [-20, 20],
                    "vy": [-20, 20],
                },
                "absolute": True,
                "flatten": False,
                "observe_intentions": False
            },
            "action": {
                "type": "ContinuousAction",
                "steering_range": [-np.pi / 3, np.pi / 3],
                "longitudinal": True,
                "lateral": True,
                "dynamical": True
            },
            "duration": 100,  # [s]
            "destination": "o1",
            "controlled_vehicles": 1,
            "screen_width": 1000,
            "screen_height": 1000,
            "centering_position": [0.5, 0.6],
            "scaling": 5.5 * 1.3,
            "show_trajectories": True,
            "policy_frequency": 10,
            "simulation_frequency": 10,
            "num_vehicles_right": 5,
            "num_vehicles_left": 5,
            "spawn_probability_right": 1,
            "spawn_probability_left": 1,
            "position_deviation_right": 0.1,
            "position_deviation_left": 0.1,
            "speed_deviation_right": 0.5,
            "speed_deviation_left": 0.5,
            "ego_vehicle_position": 10
        })
        return config

    def _reward(self, action: int) -> float:
        """Aggregated reward, for cooperative agents."""
        return 0.0

    def _rewards(self, action: int) -> Dict[Text, float]:
        """Multi-objective rewards, for cooperative agents."""
        return {'total_reward': 0.0}

    def _is_terminated(self) -> bool:
        return self.vehicle.crashed or not self.vehicle.on_road

    def _is_truncated(self) -> bool:
        """The episode is truncated if the time limit is reached."""
        return (self.time >= self.config["duration"]
                or self.controlled_vehicles[0].position[1] < -40
                or self.controlled_vehicles[0].position[0] < -40
                or self.controlled_vehicles[0].position[0] > 40)

    def _reset(self) -> None:
        self._make_road()
        self._make_vehicles()

    def _make_road(self) -> None:
        """
        Make an 4-way intersection.

        The horizontal road has the right of way. More precisely, the levels of priority are:
            - 3 for horizontal straight lanes and right-turns
            - 1 for vertical straight lanes and right-turns
            - 2 for horizontal left-turns
            - 0 for vertical left-turns

        The code for nodes in the road network is:
        (o:outer | i:inner + [r:right, l:left]) + (0:south | 1:west | 2:north | 3:east)

        :return: the intersection road
        """
        lane_width = AbstractLane.DEFAULT_WIDTH
        right_turn_radius = lane_width + 5  # [m}
        left_turn_radius = right_turn_radius + lane_width  # [m}
        outer_distance = right_turn_radius + lane_width / 2
        access_length = 50 + 50  # [m]

        net = RoadNetwork()
        n, c, s = LineType.NONE, LineType.CONTINUOUS, LineType.STRIPED
        for corner in range(4):
            angle = np.radians(90 * corner)
            is_horizontal = corner % 2
            priority = 3 if is_horizontal else 1
            rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
            # Incoming
            start = rotation @ np.array([lane_width / 2, access_length + outer_distance])
            end = rotation @ np.array([lane_width / 2, outer_distance])

            net.add_lane("o" + str(corner), "ir" + str(corner),
                         StraightLane(start, end, line_types=[s, c], priority=priority, speed_limit=10))

            # Right turn
            r_center = rotation @ (np.array([outer_distance, outer_distance]))

            net.add_lane("ir" + str(corner), "il" + str((corner - 1) % 4),
                         CircularLane(r_center, right_turn_radius, angle + np.radians(180), angle + np.radians(270),
                                      line_types=[n, c], priority=priority, speed_limit=10))
            # Left turn
            l_center = rotation @ (np.array([-left_turn_radius + lane_width / 2, left_turn_radius - lane_width / 2]))

            net.add_lane("ir" + str(corner), "il" + str((corner + 1) % 4),
                         CircularLane(l_center, left_turn_radius, angle + np.radians(0), angle + np.radians(-90),
                                      clockwise=False, line_types=[n, n], priority=priority - 1, speed_limit=10))
            # Straight
            start = rotation @ np.array([lane_width / 2, outer_distance])
            end = rotation @ np.array([lane_width / 2, -outer_distance])

            net.add_lane("ir" + str(corner), "il" + str((corner + 2) % 4),
                         StraightLane(start, end, line_types=[n, n], priority=priority, speed_limit=10))
            # Exit
            start = rotation @ np.flip([lane_width / 2, access_length + outer_distance], axis=0)
            end = rotation @ np.flip([lane_width / 2, outer_distance], axis=0)

            net.add_lane("il" + str((corner - 1) % 4), "o" + str((corner - 1) % 4),
                         StraightLane(end, start, line_types=[n, c], priority=priority, speed_limit=10))

        road = RegulatedRoad(network=net, np_random=self.np_random, record_history=self.config["show_trajectories"])

        # Add stop signs
        stop_sign = StopSign(road, net.get_lane(("o0", "ir0", 0)).position(95, 0))
        road.objects.append(stop_sign)
        self.road = road

    def _make_vehicles(self) -> None:
        """
        Populate a road with several vehicles on the highway and on the merging lane

        :return: the ego-vehicle
        """
        # Configure vehicles
        # vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle_type = self.action_type.vehicle_class

        vehicle_type.DISTANCE_WANTED = 7  # Low jam distance
        vehicle_type.COMFORT_ACC_MAX = 6
        vehicle_type.COMFORT_ACC_MIN = -3

        # Random vehicles
        simulation_steps = 3
        num_vehicles_right = self.config["num_vehicles_right"]
        num_vehicles_left = self.config["num_vehicles_left"]
        spawn_probability_right = self.config["spawn_probability_right"]
        spawn_probability_left = self.config["spawn_probability_left"]
        position_deviation_right = self.config["position_deviation_right"]
        position_deviation_left = self.config["position_deviation_left"]
        speed_deviation_right = self.config["speed_deviation_right"]
        speed_deviation_left = self.config["speed_deviation_left"]

        for t in range(num_vehicles_right):
            self._spawn_vehicle(np.linspace(0, 100, num_vehicles_right)[t], origin='right',
                                spawn_probability=spawn_probability_right,
                                go_straight=True,
                                position_deviation=position_deviation_right,
                                speed_deviation=speed_deviation_right)
        for t in range(num_vehicles_left):
            self._spawn_vehicle(np.linspace(0, 100, num_vehicles_left)[t], origin='left',
                                spawn_probability=spawn_probability_left,
                                go_straight=True,
                                position_deviation=position_deviation_left,
                                speed_deviation=speed_deviation_left)

        for _ in range(simulation_steps):
            [(self.road.act(), self.road.step(1 / self.config["simulation_frequency"])) for _ in
             range(self.config["simulation_frequency"])]

        # Controlled vehicles
        self.controlled_vehicles = []
        ego_vehicle_position = self.config["ego_vehicle_position"]
        for ego_id in range(0, self.config["controlled_vehicles"]):
            ego_lane = self.road.network.get_lane(("o{}".format(ego_id % 4), "ir{}".format(ego_id % 4), 0))
            destination = self.config["destination"] or "o" + str(self.np_random.randint(1, 4))
            ego_vehicle = self.action_type.vehicle_class(
                self.road,
                ego_lane.position(ego_vehicle_position, 0),
                speed=ego_lane.speed_limit,
                heading=ego_lane.heading_at(60))
            try:
                ego_vehicle.plan_route_to(destination)
                ego_vehicle.speed_index = ego_vehicle.speed_to_index(ego_lane.speed_limit)
                ego_vehicle.target_speed = ego_vehicle.index_to_speed(ego_vehicle.speed_index)
            except AttributeError:
                pass

            self.road.vehicles.append(ego_vehicle)
            self.controlled_vehicles.append(ego_vehicle)
            for v in self.road.vehicles:  # Prevent early collisions
                if v is not ego_vehicle and np.linalg.norm(v.position - ego_vehicle.position) < 20:
                    self.road.vehicles.remove(v)

    def _spawn_vehicle(self,
                       longitudinal: float = 0,
                       position_deviation: float = 1.,
                       speed_deviation: float = 1.,
                       spawn_probability: float = 0.6,
                       go_straight: bool = True,
                       origin: str = 'left') -> None:
        if self.np_random.uniform() > spawn_probability:
            return

        route = self.np_random.choice(range(4), size=2, replace=False)
        # route[1] = (route[0] + 2) % 4 if go_straight else route[1]
        if origin == 'left':
            route[0] = 1
            route[1] = 3
        elif origin == 'right':
            route[0] = 3
            route[1] = 1
        vehicle_type = utils.class_from_path(self.config["other_vehicles_type"])
        vehicle = vehicle_type.make_on_lane(self.road, ("o" + str(route[0]), "ir" + str(route[0]), 0),
                                            longitudinal=(longitudinal + 5
                                                          + self.np_random.normal() * position_deviation),
                                            speed=8 + self.np_random.normal() * speed_deviation)
        for v in self.road.vehicles:
            if np.linalg.norm(v.position - vehicle.position) < 15:
                return
        vehicle.plan_route_to("o" + str(route[1]))
        vehicle.randomize_behavior()
        self.road.vehicles.append(vehicle)
        return vehicle

    def _clear_vehicles(self) -> None:
        is_leaving = lambda vehicle: "il" in vehicle.lane_index[0] and "o" in vehicle.lane_index[1] \
                                     and vehicle.lane.local_coordinates(vehicle.position)[0] \
                                     >= vehicle.lane.length - 4 * vehicle.LENGTH
        self.road.vehicles = [vehicle for vehicle in self.road.vehicles if
                              vehicle in self.controlled_vehicles or not (is_leaving(vehicle) or vehicle.route is None)]

    def has_arrived(self, vehicle: Vehicle, exit_distance: float = 25) -> bool:
        return "il" in vehicle.lane_index[0] \
            and "o" in vehicle.lane_index[1] \
            and vehicle.lane.local_coordinates(vehicle.position)[0] >= exit_distance
