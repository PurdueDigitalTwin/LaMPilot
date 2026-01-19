import time
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium.wrappers import RecordVideo

from highway_env.envs import AbstractEnv
from highway_env.road.lane import LineType
from projects.lampilot.dt.vehicle_dt import VehicleDigitalTwin
import projects.lampilot.utils as U


class DbLEvaluator:
    def __init__(self,
                 config: dict,
                 show_window: bool = True,
                 record_video: bool = False,
                 wait_time: float = 0.0,
                 safe_ttc_threshold: float = 2.0,
                 speed_std_threshold: float = 10.0,
                 time_threshold: float = 60.0,
                 score_weights: dict = None,
                 video_dir: str = ""
                 ):
        self.exp_time = time.strftime("%Y%m%d-%H%M%S")
        self.config: dict = config
        self.record_video = record_video
        self.show_window = show_window
        self.wait_time = wait_time
        self.video_dir = video_dir

        self.safe_ttc_threshold = safe_ttc_threshold
        self.speed_std_threshold = speed_std_threshold
        self.time_threshold = time_threshold
        self.score_weights = score_weights or {
            'ttc': 0.5,
            'speed_variance': 0.3,
            'time_efficiency': 0.2,
        }

        self._init_env(config)
        self.simulation_frequency = self.env.unwrapped.config['simulation_frequency']
        self.frame = 0
        self.queue = deque(maxlen=10000)

        self._init_ego_vehicle()

        self.done = self.truncated = False
        self.success = False
        self.collision = False

    def _init_env(self, config: dict):
        # noinspection PyTypeChecker
        self.env: AbstractEnv = gym.make(config['env']['type'], render_mode="rgb_array")
        self.env.unwrapped.configure(config['env'])

        if self.record_video:
            self.env = RecordVideo(self.env, self.video_dir, name_prefix=f'{self.exp_time}', )
            self.env.unwrapped.set_record_video_wrapper(self.env)

        self.env.reset(seed=config['seed'])
        for _ in range(3 * self.env.unwrapped.config['simulation_frequency']):  # warm up
            self.env.step(np.array([0., 0.]))

    def _init_ego_vehicle(self):
        self.ego_vehicle = self.env.unwrapped.vehicle
        self.ego_vehicle.speed = 20
        speed_limit = self.ego_vehicle.road.network.get_lane(self.ego_vehicle.lane_index).speed_limit
        if speed_limit:
            self.ego_vehicle.speed = speed_limit
        front_vehicle = self.ego_vehicle.road.neighbour_vehicles(self.ego_vehicle, self.ego_vehicle.lane_index)[0]
        if front_vehicle is not None:
            if self.ego_vehicle.lane_distance_to(front_vehicle) < 50:
                self.ego_vehicle.speed = front_vehicle.speed

    def step(self, agent: VehicleDigitalTwin):
        if self.show_window:
            self.env.render()
            time.sleep(self.wait_time)
        action = agent.act()
        _, _, self.done, self.truncated, info = self.env.step(action)
        self.collision = self.ego_vehicle.crashed or not self.ego_vehicle.on_road

        self._append({
            'acceleration': action[0],
            'steering': action[1],
            'speed': agent.speed,
            'ttc': U.compute_ttc(self.env)
        })
        self.frame += 1

    def close(self):
        self.env.close()

    @property
    def ended(self) -> bool:
        return self.done or self.truncated

    @staticmethod
    def human_check_task_success():
        confirmed = False
        success = False
        commit = False
        critique = ""
        while not confirmed:
            success = input("Success? (y/n)")
            success = success.lower() == "y"
            if success:
                commit = input("Commit? (y/n)")
                commit = commit.lower() == "y"
            critique = input("Enter your critique:")
            print(f"Success: {success}\nCommit: {commit}\nCritique: {critique}")
            confirmed = input("Confirm? (y/n)") in ["y", ""]
        return success, commit, critique

    def check_task_success(self):
        return self.human_check_task_success()

    @property
    def score(self) -> float:
        if not self.success:
            return 0
        else:
            return max(0., self.score_ttc * self.score_weights['ttc']
                       + self.score_speed_variance * self.score_weights['speed_variance']
                       + self.score_time_efficiency * self.score_weights['time_efficiency'])

    @property
    def score_ttc(self) -> float:
        ttcs = [item['ttc'] for item in self.queue if item['ttc'] > 0]
        if len(ttcs) == 0:  # no conflict
            return 100.
        else:
            min_ttc = min(ttcs)
            if min_ttc > self.safe_ttc_threshold:  # safe
                return 100
            else:
                return max(100 - (1 / min_ttc), -100.)

    @property
    def score_speed_variance(self) -> float:
        return 100 * (1 - self.speed_std / self.speed_std_threshold)

    @property
    def score_time_efficiency(self) -> float:
        return 100 * (1 - self.overall_time / self.time_threshold)

    @property
    def overall_time(self) -> float:
        return self.frame / self.simulation_frequency

    @property
    def speed_std(self) -> float:
        return np.std([item['speed'] for item in self.queue]).item()

    def _append(self, item: dict):
        self.queue.append(item)

    @property
    def _lanes(self):
        _from, _to, _id = self.ego_vehicle.lane_index
        n = int(len(self.ego_vehicle.road.network.graph[_from][_to]))
        right_count = n - _id

        right_most_lane_index = (_from, _to, n - 1)
        if self.ego_vehicle.road.network.get_lane(right_most_lane_index).line_types == (
                LineType.CONTINUOUS_LINE, LineType.CONTINUOUS_LINE):
            right_emergency_lane = True
        else:
            right_emergency_lane = False
        return n, right_count, right_emergency_lane

    @property
    def _front_vehicle(self):
        front = self.ego_vehicle.road.neighbour_vehicles(self.ego_vehicle, self.ego_vehicle.lane_index)[0]
        if front:
            front_distance = self.ego_vehicle.lane_distance_to(front)
            if front_distance < 100:
                front_speed = front.speed
                return front, front_distance, front_speed

        return None, None, None

    def get_context_info(self) -> str:
        if self.config['env']['type'] in ['ramp-merge-v0', 'dt-highway-v0']:
            f, fd, fs = self._front_vehicle
            n, rc, re = self._lanes

            context = f"My current speed is {self.ego_vehicle.speed:.1f} m/s. "
            context += f"I am driving on a highway with {n:d} lanes in my direction, " \
                       f"and I am in the {U.ordinal(rc)} lane from the right. "
            if re:
                context += "The right-most lane is an emergency lane. "
            if f:
                context += f"There is a car in front of me in my lane, " \
                           f"at a distance of {fd:.1f} m, with a speed of {fs:.1f} m/s. "
            else:
                context += "There is no car ahead of me in my lane. "
            return context

        elif self.config['env']['type'] in ['dt-intersection-v0']:
            context = f"My current speed is {self.ego_vehicle.speed:.1f} m/s. "
            context += "I am driving on a two-way road with one lane in each direction. "
            context += "I am approaching an intersection with a stop sign that is 90 meters ahead. " \
                       "and the `autopilot` will bring the car to a stop. "
            context += "The current lane allows for left turns, right turns, and going straight. "
            context += "In case you want to change the planned route, do it now before the car stops. "
            context += "Continuously monitor the speed and only recover from stop " \
                       "when the speed has fallen below 1 m/s and it is safe to enter the cross traffic lanes. "
            return context

        else:
            raise RuntimeError(f"Unsupported env type {self.config['env']['type']}")
