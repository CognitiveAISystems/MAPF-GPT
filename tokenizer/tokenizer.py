import numpy as np
import torch

from tokenizer.parameters import InputParameters


class Tokenizer:
    def __init__(self, cfg: InputParameters) -> None:
        self.encoder = Encoder(cfg)

    def __call__(self, obs, return_tensors="pt"):
        assert return_tensors == "pt", "Only pt (PyTorch) encoded tensor is supported"
        idx = self.encoder.encode(obs)
        out = torch.tensor(idx, dtype=torch.int8)
        return out

    def encode(self, obs):
        idx = self.encoder.encode(obs)
        out = np.array(idx, dtype=np.int8)
        return out

    def decode(self, idx):
        assert idx.ndim == 1
        obs = self.encoder.decode(idx.tolist())
        return obs


class Encoder:
    def __init__(self, cfg: InputParameters):
        self.cfg = cfg
        self.coord_range = list(
            range(-cfg.cost2go_value_limit, cfg.cost2go_value_limit + 1)
        ) + [
            -cfg.cost2go_value_limit * 4,
            -cfg.cost2go_value_limit * 2,
            cfg.cost2go_value_limit * 2,
        ]
        self.actions_range = ["n", "w", "u", "d", "l", "r"]
        self.next_action_range = [format(i, "04b") for i in range(16)]  # 0000 to 1111

        self.vocab = {
            token: idx
            for idx, token in enumerate(
                self.coord_range + self.actions_range + self.next_action_range + ["!"]
            )
        }  # '!' is a trash symbol
        self.inverse_vocab = {idx: token for token, idx in self.vocab.items()}

    def encode(self, observation):
        agents_indices = []

        def clamp_value(value, max_abs_value=20):
            return max(-max_abs_value, min(max_abs_value, value))

        for agent in observation["agents"]:
            coord_indices = [
                self.vocab[clamp_value(agent["relative_pos"][0])],
                self.vocab[clamp_value(agent["relative_pos"][1])],
                self.vocab[clamp_value(agent["relative_goal"][0])],
                self.vocab[clamp_value(agent["relative_goal"][1])],
            ]
            actions_indices = [
                self.vocab[action] for action in agent["previous_actions"]
            ]
            next_action_indices = [self.vocab[agent["next_action"]]]

            agent_obs = coord_indices + actions_indices + next_action_indices
            agents_indices.extend(agent_obs)
        if len(observation["agents"]) < self.cfg.num_agents:
            agents_indices.extend(
                [
                    self.vocab["!"]
                    for _ in range(
                        (self.cfg.num_agents - len(observation["agents"]))
                        * (5 + self.cfg.num_previous_actions)
                    )
                ]
            )
        cost2go_indices = [
            self.vocab[v] for v in np.array(observation["cost2go"]).flatten()
        ]

        result = (
            cost2go_indices
            + agents_indices
            + [
                self.vocab["!"]
                for _ in range(
                    self.cfg.context_size - len(cost2go_indices) - len(agents_indices)
                )
            ]
        )
        if any(
            [
                self.cfg.mask_actions_history,
                self.cfg.mask_cost2go,
                self.cfg.mask_goal,
                self.cfg.mask_greed_action,
            ]
        ):
            result = self.mask(result)
        return result

    def mask(self, input):
        cost2go_size = (self.cfg.cost2go_radius * 2 + 1) ** 2
        if self.cfg.mask_actions_history:
            for i in range(self.cfg.num_agents):
                input[
                    cost2go_size
                    + i * (5 + self.cfg.num_previous_actions)
                    + 4 : cost2go_size
                    + i * (5 + self.cfg.num_previous_actions)
                    + 4
                    + self.cfg.num_previous_actions
                ] = [self.vocab["!"] for _ in range(self.cfg.num_previous_actions)]
        if self.cfg.mask_cost2go:
            traversable_cell = self.vocab[0]
            blocked_cell = self.vocab[-self.cfg.cost2go_value_limit * 4]
            for i in range(cost2go_size):
                if input[i] != blocked_cell:
                    input[i] = traversable_cell
        if self.cfg.mask_goal:
            for i in range(self.cfg.num_agents):
                input[cost2go_size + i * (5 + self.cfg.num_previous_actions) + 2] = (
                    self.vocab["!"]
                )
                input[cost2go_size + i * (5 + self.cfg.num_previous_actions) + 3] = (
                    self.vocab["!"]
                )
        if self.cfg.mask_greed_action:
            for i in range(self.cfg.num_agents):
                input[
                    cost2go_size
                    + i * (5 + self.cfg.num_previous_actions)
                    + 4
                    + self.cfg.num_previous_actions
                ] = self.vocab["!"]
        return input

    def decode(self, idx):
        if any(
            [
                self.cfg.mask_actions_history,
                self.cfg.mask_cost2go,
                self.cfg.mask_goal,
                self.cfg.mask_greed_action,
            ]
        ):
            idx = self.mask(idx)
        agents_info_size = 4 + self.cfg.num_previous_actions + 1
        cost2go_size = (self.cfg.cost2go_radius * 2 + 1) ** 2
        agents = []
        for i in range(self.cfg.num_agents):
            agent_indices = idx[
                cost2go_size
                + i * agents_info_size : cost2go_size
                + (i + 1) * agents_info_size
            ]

            relative_pos = (
                self.inverse_vocab[agent_indices[0]],
                self.inverse_vocab[agent_indices[1]],
            )
            relative_goal = (
                self.inverse_vocab[agent_indices[2]],
                self.inverse_vocab[agent_indices[3]],
            )
            previous_actions = [self.inverse_vocab[a] for a in agent_indices[4:-1]]
            next_action = self.inverse_vocab[agent_indices[-1]]

            agent = {
                "relative_pos": relative_pos,
                "relative_goal": relative_goal,
                "previous_actions": previous_actions,
                "next_action": next_action,
            }
            agents.append(agent)

        cost2go_indices = idx[:cost2go_size]
        cost2go = [self.inverse_vocab[v] for v in cost2go_indices]
        cost2go_size = self.cfg.cost2go_radius * 2 + 1
        cost2go = np.array(cost2go).reshape(cost2go_size, cost2go_size)
        observation = {"agents": agents, "cost2go": cost2go}

        return observation
