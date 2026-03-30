from gymnasium import Wrapper
from loguru import logger
from pogema import AnimationConfig, AnimationMonitor, pogema_v0
from pogema.wrappers.metrics import AgentsDensityWrapper, RuntimeMetricWrapper
from pogema_toolbox.create_env import MultiMapWrapper


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

        return observations, rewards, terminated, truncated, infos

    def reset(self, **kwargs):
        observations, info = self.env.reset(**kwargs)
        self.made_actions = [[] for _ in observations]
        self.init_positions = [obs["global_xy"] for obs in observations]
        if self.unwrapped.grid_config.on_target == "restart":
            self.global_lifelong_targets_xy = [
                [[int(x), int(y)] for x, y in obs["global_lifelong_targets_xy"]]
                for obs in observations
            ]
        return observations, info


def create_eval_env(config):
    env = pogema_v0(grid_config=config)
    env = AgentsDensityWrapper(env)
    env = MultiMapWrapper(env)
    env = RuntimeMetricWrapper(env)

    if config.with_animation:
        logger.debug("Wrapping environment with AnimationMonitor")
        env = AnimationMonitor(env, AnimationConfig(save_every_idx_episode=None))

    return env


def create_logging_env(config):
    env = pogema_v0(grid_config=config)
    env = AgentsDensityWrapper(env)
    env = MultiMapWrapper(env)
    env = LogActions(env)
    if config.with_animation:
        logger.debug("Wrapping environment with AnimationMonitor")
        env = AnimationMonitor(env, AnimationConfig(save_every_idx_episode=None))

    env = RuntimeMetricWrapper(env)

    return env
