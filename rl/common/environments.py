#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Type

import habitat
from habitat import Config, Dataset
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.utils.visualizations import maps
import numpy as np
import random

@baseline_registry.register_env(name="ExpRLEnv")
class ExpRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG

        self._previous_action = None
        self._episode_distance_covered = None
        super().__init__(self._core_env_config, dataset)
        # prepare to generate random sample for traning
        self._get_navigable_coord()


    def _get_navigable_coord(self):
        # get the navigable coord
        coordinate_max = maps.COORDINATE_MAX
        coordinate_min = maps.COORDINATE_MIN
        # to low the memory requirement
        self.top_down_grid_size = self._rl_config.ANS.MAPPER.map_scale*2.5
        resolution = (coordinate_max - coordinate_min) / self.top_down_grid_size
        grid_resolution = (int(resolution), int(resolution))

        top_down_map = maps.get_topdown_map(
            self.habitat_env.sim, grid_resolution, 20000, draw_border=False,
        )
        map_w, map_h = top_down_map.shape
        intervals = (max(int(1 / self.top_down_grid_size), 1), max(int(0.5 / self.top_down_grid_size), 1))
        x_vals = np.arange(0, map_w, intervals[0], dtype=int)
        y_vals = np.arange(0, map_h, intervals[1], dtype=int)
        coors = np.stack(np.meshgrid(x_vals, y_vals), axis=2)  # (H, W, 2)
        coors = coors.reshape(-1, 2)  # (H*W, 2)
        map_vals = top_down_map[coors[:, 0], coors[:, 1]]
        self.valid_coors = coors[map_vals > 0] * self.top_down_grid_size



    def reset(self):
        self._previous_action = None

        observations = super().reset()

        return observations

    def step(self, *args, **kwargs):
        self._previous_action = kwargs["action"]
        return super().step(*args, **kwargs)

    def get_reward_range(self):
        return (
            -1.0,
            +1.0,
        )

    def get_reward(self, observations):
        reward = 0
        return reward

    def get_done(self, observations):
        done = False
        if self._env.episode_over:
            done = True
        return done

    def get_info(self, observations):
        metrics = self.habitat_env.get_metrics()
        episode_statistics = {
            "episode_id": self.habitat_env.current_episode.episode_id,
            "scene_id": self.habitat_env.current_episode.scene_id,
        }
        metrics["episode_statistics"] = episode_statistics
        return metrics

    def get_obs_at(self, *args, **kwargs):
        """[summary]

        Returns:
            observations: the observations from all the sensors 
        """
        sim_obs = self.habitat_env.sim.get_observations_at(*args,**kwargs)
        return sim_obs

    def sample_navigable_point(self):
        return self.habitat_env.sim.sample_navigable_point()

    def get_random_pos(self):
        coordinate_max = maps.COORDINATE_MAX
        coordinate_min = maps.COORDINATE_MIN
        start_y = self.habitat_env.sim.get_agent_state().position[1]
        sampled_coor = random.choice(self.valid_coors)
        position = [
                coordinate_max - sampled_coor[0].item(),
                start_y.item(),
                coordinate_min + sampled_coor[1].item(),
            ]
        heading = random.uniform(0,1)* 2 *np.pi
        rotation = [
                0.0,
                np.sin(heading / 2).item(),
                0.0,
                np.cos(heading / 2).item(),
            ]
        return {"position":position, "rotation":rotation, "keep_agent_at_new_pose":True }