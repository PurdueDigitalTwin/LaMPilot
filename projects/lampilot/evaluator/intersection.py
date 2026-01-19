from .base import *
from highway_env.road.road import LaneIndex
from highway_env.envs.merge_env import *


class IntersectionEval(DbLEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_lane_index = tuple(self.config['eval']['direction'])
        road = self.ego_vehicle.road
        self.target_lane = road.network.get_lane(self.target_lane_index)

    def _init_env(self, config: dict):
        # noinspection PyTypeChecker
        self.env: AbstractEnv = gym.make(config['env']['type'], render_mode="rgb_array")
        self.env.unwrapped.configure(config['env'])

        if self.record_video:
            self.env = RecordVideo(self.env, f'projects/lampilot/videos/', name_prefix=f'{self.exp_time}')
            self.env.unwrapped.set_record_video_wrapper(self.env)

        self.env.reset(seed=config['seed'])

    def step(self, agent: VehicleDigitalTwin):
        super().step(agent)
        ego_heading = self.ego_vehicle.heading
        lane_coords = self.target_lane.local_coordinates(self.ego_vehicle.position)
        lane_heading = self.target_lane.heading_at(lane_coords[0])

        if (self.ego_vehicle.lane_index == self.target_lane_index and
                (np.abs(ego_heading - lane_heading) < np.deg2rad(0.1) or
                 np.isclose(np.abs(ego_heading - lane_heading), 2 * np.pi, atol=1e-3))):
            self.done = True
            self.success = True
