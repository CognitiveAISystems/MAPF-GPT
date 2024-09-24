import enum
import time
from copy import deepcopy

import numpy as np
from gymnasium import Wrapper
from loguru import logger
from pogema import AnimationConfig, AnimationMonitor, pogema_v0
from pogema.generator import generate_from_possible_targets, generate_new_target
from pogema.wrappers.metrics import AgentsDensityWrapper, RuntimeMetricWrapper
from pogema_toolbox.create_env import MultiMapWrapper


class ProvideFutureTargetsWrapper(Wrapper):
    def _get_lifelong_global_targets_xy(self):
        all_goals = []
        cur_goals = self.grid.get_targets_xy()
        generators = deepcopy(self.random_generators)
        for agent_idx in range(self.grid_config.num_agents):
            distance = 0
            cur_goal = cur_goals[agent_idx]
            goals = [cur_goal]
            while distance < self.grid_config.max_episode_steps + 100:
                if self.grid_config.possible_targets_xy is None:
                    new_goal = generate_new_target(
                        generators[agent_idx],
                        self.grid.point_to_component,
                        self.grid.component_to_points,
                        cur_goal,
                    )
                else:
                    new_goal = generate_from_possible_targets(
                        generators[agent_idx],
                        self.grid_config.possible_targets_xy,
                        cur_goal,
                    )
                    new_goal = (
                        new_goal[0] + self.grid_config.obs_radius,
                        new_goal[1] + self.grid_config.obs_radius,
                    )
                distance += abs(cur_goal[0] - new_goal[0]) + abs(
                    cur_goal[1] - new_goal[1]
                )
                cur_goal = new_goal
                goals.append(cur_goal)
            all_goals.append(goals)
        return all_goals

    def reset(self, **kwargs):
        observations, infos = self.env.reset(seed=self.env.grid_config.seed)
        observations[0]["after_reset"] = True
        observations[0]["max_episode_steps"] = self.env.grid_config.max_episode_steps
        if self.env.grid_config.on_target == "restart":
            global_lifelong_targets_xy = self._get_lifelong_global_targets_xy()
            for idx, obs in enumerate(observations):
                obs["global_lifelong_targets_xy"] = global_lifelong_targets_xy[idx]
        return observations, infos


class LogActions(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.made_actions = None
        self.init_positions = None

    def step(self, actions):
        observations, rewards, terminated, truncated, infos = self.env.step(actions)
        for i, action in enumerate(actions):
            self.made_actions[i].append(action)
        if all(terminated) or all(truncated):
            infos[0]["metrics"]["made_actions"] = self.made_actions
            infos[0]["metrics"]["init_positions"] = self.init_positions
            if self.env.grid_config.on_target == "restart":
                infos[0]["metrics"][
                    "global_lifelong_targets_xy"
                ] = self.global_lifelong_targets_xy
        return observations, rewards, terminated, truncated, infos

    def reset(self, **kwargs):
        observations, info = self.env.reset(**kwargs)
        self.made_actions = [[] for _ in observations]
        self.init_positions = [obs["global_xy"] for obs in observations]
        if self.env.grid_config.on_target == "restart":
            self.global_lifelong_targets_xy = [
                [[int(x), int(y)] for x, y in obs["global_lifelong_targets_xy"]]
                for obs in observations
            ]
        return observations, info


def create_eval_env(config):
    env = pogema_v0(grid_config=config)
    env = AgentsDensityWrapper(env)
    env = MultiMapWrapper(env)
    if config.with_animation:
        logger.debug("Wrapping environment with AnimationMonitor")
        env = AnimationMonitor(env, AnimationConfig(save_every_idx_episode=None))
    env = RuntimeMetricWrapper(env)
    return env


def create_logging_env(config):
    env = pogema_v0(grid_config=config)
    env = AgentsDensityWrapper(env)
    env = ProvideFutureTargetsWrapper(env)
    env = MultiMapWrapper(env)
    env = LogActions(env)
    if config.with_animation:
        logger.debug("Wrapping environment with AnimationMonitor")
        env = AnimationMonitor(env, AnimationConfig(save_every_idx_episode=None))

    # Adding runtime metrics
    env = RuntimeMetricWrapper(env)

    return env
