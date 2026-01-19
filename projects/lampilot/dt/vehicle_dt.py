from abc import ABC, abstractmethod
from typing import List

import numpy as np
from langchain.pydantic_v1 import BaseModel, Field

from highway_env.road.road import LaneIndex, Route
from highway_env.utils import not_zero, wrap_to_pi
from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import ControlledVehicle
from highway_env.vehicle.kinematics import Vehicle
from projects.lampilot.vehicle.objects import StopSign


class VehicleConfig(BaseModel):
    kp_a: float = Field(
        description="longitudinal speed control gain",
        default=ControlledVehicle.KP_A
    )
    kp_lat: float = Field(
        description="lateral position control gain",
        default=ControlledVehicle.KP_LATERAL
    )
    kp_psi: float = Field(
        description="heading control gain",
        default=ControlledVehicle.KP_HEADING
    )
    acceleration: float = Field(
        description="IDM parameter in m/s^2; the desired maximum vehicle acceleration",
        default=6.0)
    comfortable_deceleration: float = Field(
        description="IDM parameter in m/s^2; a positive number",
        default=6.0)
    acceleration_exponent: float = Field(description="IDM parameter", default=4.0)
    desired_time_headway: float = Field(
        description="IDM parameter in s; the minimum possible time to the vehicle in front",
        default=1.5)  # 0.5
    minimum_spacing: float = Field(
        description="IDM parameter in m; a minimum desired net distance, "
                    "a car can't move if the distance from the car in the front is not at least this value",
        default=6.0)


class VehicleDigitalTwin(ABC):
    PERCEPTION_RANGE = 100

    def __init__(self, ego_vehicle: Vehicle):
        self.ego_vehicle: Vehicle = ego_vehicle

    @abstractmethod
    def act(self) -> np.ndarray:
        pass

    @abstractmethod
    def execute(self, policy: str):
        pass

    @property
    def speed(self) -> float:
        return self.ego_vehicle.speed


class IDMDT(VehicleDigitalTwin):
    TAU_PURSUIT = ControlledVehicle.TAU_PURSUIT
    MAX_STEERING_ANGLE = ControlledVehicle.MAX_STEERING_ANGLE

    def __init__(self,
                 ego_vehicle: Vehicle = None,
                 ):
        super().__init__(ego_vehicle)
        if ego_vehicle is not None:
            self.reset(ego_vehicle)

    def reset(self, ego_vehicle: Vehicle):
        self.ego_vehicle = ego_vehicle
        self.vehicle_cfg = VehicleConfig()
        self.target_speed: float = self.ego_vehicle.speed
        self.target_lane_index: LaneIndex = self.ego_vehicle.lane_index
        self.route: Route = []
        self._ignored_stop_signs = set([])

    def act(self) -> np.ndarray:
        self._follow_road()
        action = {
            "steering": self._steering_control(self.target_lane_index),
            "acceleration": self._intelligent_driver_model(self.target_speed)
        }
        return np.array([action["acceleration"], action["steering"]])

    def execute(self, policy: str):
        pass

    def _intelligent_driver_model(self, target_speed: float, ego: Vehicle = None, front: Vehicle = None):
        desired_accel_max = self.vehicle_cfg.acceleration
        delta = self.vehicle_cfg.acceleration_exponent

        ego = ego or self.ego_vehicle
        # front = front or ego.road.neighbour_vehicles(ego, ego.lane_index)[0]
        front = front or self._front_vehicle_or_stop_sign()

        accel = desired_accel_max * (
                1 - np.power(max(ego.speed, 0) / abs(not_zero(target_speed)), delta)
        )
        if front:
            d = ego.lane_distance_to(front)
            d_star = self._compute_distance_headway(ego, front)
            if isinstance(front, StopSign):
                d_star -= self.vehicle_cfg.minimum_spacing / 2

            accel -= desired_accel_max * (
                np.power(d_star / not_zero(d), 2)
            )
        return accel

    def _compute_distance_headway(self, ego: Vehicle = None, front: Vehicle = None, time_headway: float = None):
        ego = ego or self.ego_vehicle
        front = front or ego.road.neighbour_vehicles(ego, ego.lane_index)[0]
        tau = time_headway or self.vehicle_cfg.desired_time_headway

        if front is None:
            return -1

        d0 = self.vehicle_cfg.minimum_spacing
        ab = self.vehicle_cfg.acceleration * self.vehicle_cfg.comfortable_deceleration

        dv = np.dot(ego.velocity - front.velocity, ego.direction)
        d_star = d0 + ego.speed * tau + ego.speed * dv / (2 * np.sqrt(ab))
        return d_star

    def _plan_route_to(self, destination: str):
        """
        Plan a route from the current position to a destination.
        :param destination: a node in the road network
        """
        try:
            path = self.ego_vehicle.road.network.shortest_path(self.ego_vehicle.lane_index[1], destination)
        except KeyError:
            path = []
        if path:
            self.route = [self.ego_vehicle.lane_index] + [(path[i], path[i + 1], 0) for i in range(len(path) - 1)]
        else:
            self.route = [self.ego_vehicle.lane_index]

    def _front_vehicle_or_stop_sign(self):
        ego_vehicle = self.ego_vehicle
        lane_index = ego_vehicle.lane_index
        road = ego_vehicle.road

        if not lane_index:
            return None
        lane = road.network.get_lane(lane_index)
        s = road.network.get_lane(lane_index).local_coordinates(ego_vehicle.position)[0]
        s_front = obj_front = None
        for obj in road.objects + road.vehicles:
            if (isinstance(obj, Vehicle) and obj is not ego_vehicle) or (
                    isinstance(obj, StopSign) and obj not in self._ignored_stop_signs):
                s_obj, lat_obj = lane.local_coordinates(obj.position)
                if not lane.on_lane(obj.position, s_obj, lat_obj):
                    continue
                if s <= s_obj and (s_front is None or s_obj <= s_front):
                    s_front = s_obj
                    obj_front = obj
        return obj_front

    def _front_stop_sign(self):
        ego_vehicle = self.ego_vehicle
        lane_index = ego_vehicle.lane_index
        road = ego_vehicle.road

        if not lane_index:
            return None
        lane = road.network.get_lane(lane_index)
        s = road.network.get_lane(lane_index).local_coordinates(ego_vehicle.position)[0]
        s_front = obj_front = None
        for obj in road.objects:
            if isinstance(obj, StopSign) and obj not in self._ignored_stop_signs:
                s_obj, lat_obj = lane.local_coordinates(obj.position)
                if not lane.on_lane(obj.position, s_obj, lat_obj):
                    continue
                if s <= s_obj and (s_front is None or s_obj <= s_front):
                    s_front = s_obj
                    obj_front = obj
        return obj_front

    def _follow_road(self):
        """
        At the end of lane, automatically switch to a new one
        """
        if self.ego_vehicle.road.network.get_lane(self.target_lane_index).after_end(self.ego_vehicle.position):
            target_lane_index = self.ego_vehicle.road.network.next_lane(
                self.target_lane_index,
                position=self.ego_vehicle.position,
                np_random=self.ego_vehicle.road.np_random,
                route=self.route
            )
            self.target_lane_index = target_lane_index

    def _steering_control(self, target_lane_index: LaneIndex) -> float:
        """
        Steer the vehicle to follow the center of an given lane.

        1. Lateral position is controlled by a proportional controller yielding a lateral speed command
        2. Lateral speed command is converted to a heading reference
        3. Heading is controlled by a proportional controller yielding a heading rate command
        4. Heading rate command is converted to a steering angle

        :param target_lane_index: index of the lane to follow
        :return: a steering wheel angle command [rad]
        """
        target_lane = self.ego_vehicle.road.network.get_lane(target_lane_index)
        lane_coords = target_lane.local_coordinates(self.ego_vehicle.position)
        lane_next_coords = lane_coords[0] + self.ego_vehicle.speed * self.TAU_PURSUIT
        lane_future_heading = target_lane.heading_at(lane_next_coords)

        # Position control
        delta_lat = lane_coords[1]
        lateral_velocity_command = - self.vehicle_cfg.kp_lat * delta_lat
        # Heading variation to apply the lateral velocity command
        heading_command = np.arcsin(np.clip(lateral_velocity_command / not_zero(self.ego_vehicle.speed), -1, 1))

        # Heading control
        heading_ref = lane_future_heading + np.clip(heading_command, -np.pi / 4, np.pi / 4)
        heading_rate_command = self.vehicle_cfg.kp_psi * wrap_to_pi(heading_ref - self.ego_vehicle.heading)

        # Heading rate to steering angle
        slip_angle = np.arcsin(
            np.clip(self.ego_vehicle.LENGTH / 2 / not_zero(self.ego_vehicle.speed) * heading_rate_command, -1, 1))
        steering_angle = np.arctan(2 * np.tan(slip_angle))
        steering_angle = np.clip(steering_angle, -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE)
        return float(steering_angle)

    def _speed_control(self, target_speed: float) -> float:
        """
        Control the speed of the vehicle using a proportional controller.

        :param target_speed: the speed to reach [m/s]
        :return: an acceleration command [m/s2]
        """
        return self.vehicle_cfg.kp_a * (target_speed - self.ego_vehicle.speed)


class MOBILDT(IDMDT):
    def __init__(self, ego_vehicle: Vehicle = None):
        super().__init__(ego_vehicle)
        self.politeness_factor = 0.2
        self.maximum_safe_deceleration = 10
        self.changing_threshold = 0.1

    def act(self) -> np.ndarray:
        self._follow_road()
        self._change_lane_policy()
        action = {
            "steering": self._steering_control(self.target_lane_index),
            "acceleration": self._intelligent_driver_model(self.target_speed)
        }
        return np.array([action["acceleration"], action["steering"]])

    def _change_lane_policy(self):
        for lane_index in self.ego_vehicle.road.network.side_lanes(self.ego_vehicle.lane_index):
            if self._mobil(lane_index):
                self.target_lane_index = lane_index

    def _mobil(self, lane_index: LaneIndex) -> bool:
        """
        https://traffic-simulation.de/info/info_MOBIL.html
        MOBIL = Minimizing Overall Braking decelerations Induced by Lane changes

        The vehicle should change lane only if:
        - the potential new target lane is more attractive, i.e., the incentive criterion is satisfied
        - and the change can be performed safely, i.e., the safety criterion is satisfied.
        """
        new_front: IDMVehicle
        new_back: IDMVehicle
        old_front: IDMVehicle
        old_back: IDMVehicle

        p = self.politeness_factor
        a_thr = self.changing_threshold
        b_safe = self.maximum_safe_deceleration

        new_front, new_back = self.ego_vehicle.road.neighbour_vehicles(self.ego_vehicle, lane_index)
        old_front, old_back = self.ego_vehicle.road.neighbour_vehicles(self.ego_vehicle)

        # Safety criterion
        if new_back:
            acc_prime_new_back = new_back.acceleration(ego_vehicle=new_back, front_vehicle=self.ego_vehicle)  # acc'(B')
            safety_criterion = acc_prime_new_back > -b_safe
            if not safety_criterion:
                return False

        # Incentive criterion
        acc_self = self._intelligent_driver_model(front=old_front, target_speed=self.target_speed)  # acc(M)
        acc_prime_new_self = self._intelligent_driver_model(front=new_front, target_speed=self.target_speed)  # acc'(M')
        if new_back:
            acc_new_back = new_back.acceleration(ego_vehicle=new_back, front_vehicle=new_front)  # acc(B')
        else:
            acc_new_back = acc_prime_new_back = 0

        incentive_criterion = (acc_prime_new_self - acc_self) > (p * (acc_new_back - acc_prime_new_back) + a_thr)

        if not incentive_criterion:
            return False

        return True


class CtrlVDT(IDMDT):
    TAU_PURSUIT = ControlledVehicle.TAU_PURSUIT
    MAX_STEERING_ANGLE = ControlledVehicle.MAX_STEERING_ANGLE

    def reset(self, ego_vehicle: Vehicle):
        super().reset(ego_vehicle)
        self.reset_policy()

    def execute(self, code: dict):
        apis = [
            # Ego
            "get_ego_vehicle",
            "get_desired_time_headway",
            "get_target_speed",
            "say",
            "is_safe_enter",

            # Control
            "set_desired_time_headway",
            "set_target_speed",
            "set_target_lane",
            "autopilot",
            "recover_from_stop",

            # Perception
            "get_speed_of",
            "get_lane_of",
            "detect_front_vehicle_in",
            "detect_rear_vehicle_in",
            "get_distance_between_vehicles",
            "get_left_to_right_cross_traffic_lanes",
            "get_right_to_left_cross_traffic_lanes",
            "get_left_lane",
            "get_right_lane",
            "detect_stop_sign_ahead",

            # Planning
            "turn_left_at_next_intersection",
            "go_straight_at_next_intersection",
            "turn_right_at_next_intersection",

        ]
        apis = {api: getattr(self, api) for api in apis}
        exec(code['reused_code'], apis)
        local_vars = {}
        try:
            exec(code['new_code'], apis, local_vars)
            self.policy = local_vars.get("policy", iter([]))

            if self.target_lane_index is None:
                raise ValueError("target_lane_index is None")

        except Exception as e:
            print(f"\033[31m Error in execute: {e} \033[0m")
            self.reset_policy()

    def reset_policy(self):
        if self.target_lane_index is None:
            self.target_lane_index = self.ego_vehicle.lane_index
        self.policy = iter([])

    def act(self) -> np.ndarray:
        try:
            action = next(self.policy)
            if len(action) != 2:
                raise ValueError("Invalid action")

        except StopIteration:
            self.reset_policy()
        except Exception as e:
            if "'NoneType' object is not an iterator" not in str(e):
                print(f"\033[31m Error in act: {e} \033[0m")
            self.reset_policy()
        return self.autopilot()

    # ====== API ====== #
    def turn_left_at_next_intersection(self):
        """Plan route to turn left at the next intersection."""
        self._plan_route_to("o1")

    def go_straight_at_next_intersection(self):
        """Plan route to go straight at the next intersection."""
        self._plan_route_to("o2")

    def turn_right_at_next_intersection(self):
        """Plan route to turn right at the next intersection."""
        self._plan_route_to("o3")

    @staticmethod
    def get_left_to_right_cross_traffic_lanes() -> List[LaneIndex]:
        return [('o1', 'ir1', 0), ('ir1', 'il3', 0), ('il3', 'o3', 0)]

    @staticmethod
    def get_right_to_left_cross_traffic_lanes() -> List[LaneIndex]:
        return [('il1', 'o1', 0), ('ir3', 'il1', 0), ('o3', 'ir3', 0)]

    def get_left_lane(self, vehicle: Vehicle = None, lane_index: LaneIndex = None):
        vehicle = vehicle or self.ego_vehicle
        _from, _to, _id = lane_index or vehicle.lane_index
        target_lane_index = (_from, _to, int(np.clip(_id - 1, 0, len(vehicle.road.network.graph[_from][_to]) - 1)))
        return target_lane_index if target_lane_index[2] != int(_id) else None

    def get_right_lane(self, vehicle: Vehicle = None, lane_index: LaneIndex = None):
        vehicle = vehicle or self.ego_vehicle
        _from, _to, _id = lane_index or vehicle.lane_index
        target_lane_index = (_from, _to, int(np.clip(_id + 1, 0, len(vehicle.road.network.graph[_from][_to]) - 1)))
        return target_lane_index if target_lane_index[2] != int(_id) else None

    @staticmethod
    def get_lane_of(veh: Vehicle):
        return veh.lane_index

    @staticmethod
    def get_speed_of(veh: Vehicle):
        return veh.speed

    def get_target_speed(self):
        return self.target_speed

    @staticmethod
    def get_distance_between_vehicles(veh1: Vehicle, veh2: Vehicle):
        return -veh1.lane_distance_to(veh2)

    def autopilot(self):
        self._follow_road()
        action = {
            "steering": self._steering_control(self.target_lane_index),
            "acceleration": self._intelligent_driver_model(self.target_speed)
        }
        return np.array([action["acceleration"], action["steering"]])

    def detect_front_vehicle_in(self, lane: LaneIndex, distance: float = 100):
        front, _ = self._detect_front_and_rear_vehicles_in_lane(lane)
        if front and 0 < self.ego_vehicle.lane_distance_to(front) < distance:
            return front
        return None

    def detect_rear_vehicle_in(self, lane: LaneIndex, distance: float = 100):
        _, rear = self._detect_front_and_rear_vehicles_in_lane(lane)
        if rear and -distance < self.ego_vehicle.lane_distance_to(rear) < 0:
            return rear
        return None

    def detect_stop_sign_ahead(self) -> float:
        stop_sign = self._front_stop_sign()
        if stop_sign:
            return self.ego_vehicle.lane_distance_to(stop_sign)
        else:
            return -1

    def get_ego_vehicle(self) -> Vehicle:
        return self.ego_vehicle

    def set_target_speed(self, target_speed: float):
        self.target_speed = target_speed

    def set_target_lane(self, target_lane: LaneIndex):
        try:
            if len(target_lane) != 3:
                raise ValueError("Invalid target_lane")
            self.target_lane_index = target_lane
        except Exception as e:
            print("Warning: ", e)

    def recover_from_stop(self):
        stopped_speed = 1
        if self.ego_vehicle.speed < stopped_speed:
            front = self._front_vehicle_or_stop_sign()
            if isinstance(front, StopSign):
                self._ignored_stop_signs.add(front)

    @staticmethod
    def say(text: str):
        print(f"[LLM]: {text}")

    def set_desired_time_headway(self, desired_time_headway: float):
        self.vehicle_cfg.desired_time_headway = np.clip(desired_time_headway, 0.01, 2.0).item()

    def get_desired_time_headway(self):
        return self.vehicle_cfg.desired_time_headway

    def get_speed_limit(self) -> float:
        return self.ego_vehicle.road.network.get_lane(self.ego_vehicle.lane_index).speed_limit

    def _detect_front_and_rear_vehicles_in_lane(self, lane_index: LaneIndex):
        return self.ego_vehicle.road.neighbour_vehicles(self.ego_vehicle, lane_index)

    def is_safe_enter(self, lane: LaneIndex, safe_decel: float = 5):
        if lane is None:
            return False

        front, rear = self._detect_front_and_rear_vehicles_in_lane(lane)
        ego = self.ego_vehicle
        if front:
            ego_decel = self._intelligent_driver_model(
                target_speed=self.get_speed_of(ego),
                ego=self.ego_vehicle,
                front=front
            )
            if ego_decel < -safe_decel:
                return False
        if rear:
            rear_decel = self._intelligent_driver_model(
                target_speed=self.get_speed_of(rear),
                ego=rear,
                front=self.ego_vehicle
            )
            if rear_decel < -safe_decel:
                return False
        return True

