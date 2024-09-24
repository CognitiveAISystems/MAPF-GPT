from pathlib import Path
from typing import Literal, Optional

import cppimport.import_hook
import torch
from huggingface_hub import hf_hub_download
from pogema_toolbox.algorithm_config import AlgoBase
from pogema_toolbox.registry import ToolboxRegistry
from pydantic import Extra

from gpt.model import GPT, GPTConfig
from tokenizer import cost2go
from tokenizer.tokenizer import Encoder, InputParameters


class MAPFGPTInferenceConfig(AlgoBase, extra=Extra.forbid):
    name: Literal["MAPF-GPT"] = "MAPF-GPT"
    num_agents: int = 13
    num_previous_actions: int = 5
    cost2go_value_limit: int = 20
    agents_radius: int = 5
    cost2go_radius: int = 5
    path_to_weights: Optional[str] = "weights/model-6M.pt"
    device: str = "cuda"
    context_size: int = 256
    mask_actions_history: bool = False
    mask_goal: bool = False
    mask_cost2go: bool = False
    mask_greed_action: bool = False
    repo_id: str = 'aandreychuk/MAPF-GPT'


def strip_prefix_from_state_dict(state_dict, prefix="_orig_mod."):
    """
    strips the given prefix from the keys in the state dictionary
    """
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith(prefix):
            new_key = k[len(prefix):]
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    return new_state_dict


class MAPFGPTInference:
    def __init__(self, cfg: MAPFGPTInferenceConfig, net=None):
        self.cfg: MAPFGPTInferenceConfig = cfg
        self.cost2go_data = None
        self.actions_history = None
        self.position_history = None
        self.encoder = Encoder(
            InputParameters(
                num_agents=cfg.num_agents,
                num_previous_actions=cfg.num_previous_actions,
                cost2go_value_limit=cfg.cost2go_value_limit,
                agents_radius=cfg.agents_radius,
                cost2go_radius=cfg.cost2go_radius,
                context_size=cfg.context_size,
                mask_actions_history=cfg.mask_actions_history,
                mask_cost2go=cfg.mask_cost2go,
                mask_goal=cfg.mask_goal,
                mask_greed_action=cfg.mask_greed_action,
            )
        )

        path_to_weights = Path(self.cfg.path_to_weights)
        if path_to_weights.name in ['model-2M.pt', 'model-6M.pt', 'model-85M.pt']:
            hf_hub_download(repo_id=self.cfg.repo_id, filename=path_to_weights.name, local_dir=path_to_weights.parent)
            ToolboxRegistry.info(f'Using weights loaded from huggingface: {path_to_weights}')

        if self.cfg.device in ['mps', 'cuda'] and not torch.cuda.is_available() if self.cfg.device == 'cuda' else not torch.backends.mps.is_available():
            ToolboxRegistry.warning(f'{self.cfg.device} is not available, using cpu instead!')
            self.cfg.device = 'cpu'

        checkpoint = torch.load(
            Path(self.cfg.path_to_weights), map_location=self.cfg.device
        )

        model_state_dict = strip_prefix_from_state_dict(checkpoint["model"])
        config_dict = checkpoint.get("model_args")
        gpt_config = GPTConfig(**config_dict)
        if net is not None:
            self.net = net
        else:
            self.net = GPT(gpt_config)
            self.net.load_state_dict(model_state_dict, strict=False)
            self.net.to(self.cfg.device)
            self.net.eval()

    def generate_input(self, observations):
        next_actions = ["" for _ in range(len(observations))]
        for agent_idx, obs in enumerate(observations):
            next_action = ""
            for m in [[-1, 0], [1, 0], [0, -1], [0, 1]]:
                new_pos = (obs["global_xy"][0] + m[0], obs["global_xy"][1] + m[1])
                if (
                    self.cost2go_data[obs["global_target_xy"]][new_pos[0]][new_pos[1]]
                    >= 0
                    and self.cost2go_data[obs["global_target_xy"]][obs["global_xy"][0]][
                        obs["global_xy"][1]
                    ]
                    > self.cost2go_data[obs["global_target_xy"]][new_pos[0]][new_pos[1]]
                ):
                    next_action += "1"
                else:
                    next_action += "0"
            next_actions[agent_idx] = next_action

        inputs = []
        global_xy = [obs["global_xy"] for obs in observations]

        for agent_idx, obs in enumerate(observations):
            agents_info = []
            distances = []
            for j, p2 in enumerate(global_xy):
                distance = self.cost2go_data[tuple(global_xy[agent_idx])][p2[0]][p2[1]]
                if distance >= 0:
                    distances.append((j, distance))
            distances.sort(key=lambda x: (x[1], x[0]))
            sorted_agents = [agent_id for agent_id, _ in distances]
            for n in sorted_agents[: self.cfg.num_agents]:
                relative_goal = (
                    observations[n]["global_target_xy"][0] - obs["global_xy"][0],
                    observations[n]["global_target_xy"][1] - obs["global_xy"][1],
                )
                relative_xy = (
                    observations[n]["global_xy"][0] - obs["global_xy"][0],
                    observations[n]["global_xy"][1] - obs["global_xy"][1],
                )
                if -self.cfg.agents_radius <= relative_xy[0] <= self.cfg.agents_radius \
                    and -self.cfg.agents_radius <= relative_xy[1] <= self.cfg.agents_radius:
                    agents_info.append(
                        {
                            "relative_pos": relative_xy,
                            "relative_goal": relative_goal,
                            "previous_actions": self.actions_history[n],
                            "next_action": next_actions[n],
                        }
                    )
            inputs.append(
                {
                    "agents": agents_info,
                    "cost2go": cost2go.generate_cost2go_obs(
                        self.cost2go_data[obs["global_target_xy"]],
                        obs["global_xy"],
                        self.cfg.cost2go_radius,
                        self.cfg.cost2go_value_limit,
                        self.cfg.mask_cost2go,
                    ),
                }
            )

        return inputs

    def act(self, observations):
        num_agents = len(observations)
        moves = {(0, 0): "w", (-1, 0): "u", (1, 0): "d", (0, -1): "l", (0, 1): "r"}
        if self.cost2go_data is None:
            global_obs = observations[0]["global_obstacles"].copy().astype(int).tolist()
            self.cost2go_data = cost2go.precompute_cost2go(
                global_obs, self.cfg.cost2go_radius
            )
            self.actions_history = [["n" for _ in range(self.cfg.num_previous_actions)] for _ in range(num_agents)]
            self.position_history = [[obs['global_xy']] for obs in observations]
        else:
            for i in range(num_agents):
                self.position_history[i].append(observations[i]["global_xy"])
                self.actions_history[i].append(moves[(self.position_history[i][-1][0] - self.position_history[i][-2][0], 
                                                      self.position_history[i][-1][1] - self.position_history[i][-2][1])])
                self.actions_history[i] = self.actions_history[i][-self.cfg.num_previous_actions:]
        inputs = self.generate_input(observations)
        tensor_obs = torch.tensor(
            [self.encoder.encode(input) for input in inputs],
            dtype=torch.long,
            device=self.cfg.device,
        )

        actions = torch.squeeze(self.net.act(tensor_obs)).tolist()
        if not isinstance(actions, list):
            actions = [actions]
        return actions

    def reset_states(self):
        self.cost2go_data = None
        self.actions_history = None
        self.position_history = None
