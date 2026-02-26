from pathlib import Path
from typing import Literal, Optional

import cppimport.import_hook # noqa: F401
import torch
from huggingface_hub import hf_hub_download
from pogema_toolbox.algorithm_config import AlgoBase
from pogema_toolbox.registry import ToolboxRegistry
from pydantic import Extra

from mapf_gpt.model import GPT, GPTConfig
from mapf_gpt.observation_generator import ObservationGenerator, InputParameters


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
    grid_step: int = 64
    save_cost2go: bool = False
    batch_size: int = 2048
    num_process: int = 8

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
        self.input_parameters = InputParameters(
            cfg.cost2go_value_limit,
            cfg.num_agents,
            cfg.num_previous_actions,
            cfg.context_size,
            cfg.cost2go_radius,
            cfg.agents_radius,
            cfg.grid_step,
            cfg.save_cost2go
        )
        self._obs_generators = {}
        self._last_actions = {}

        path_to_weights = Path(self.cfg.path_to_weights)
        if path_to_weights.name in ['model-2M.pt', 'model-6M.pt', 'model-85M.pt']:
            hf_hub_download(repo_id=self.cfg.repo_id, filename=path_to_weights.name, local_dir=path_to_weights.parent)
            ToolboxRegistry.info(f'Using weights loaded from huggingface: {path_to_weights}')

        if ('cuda' in self.cfg.device and not torch.cuda.is_available()) or (self.cfg.device == 'mps' and not torch.backends.mps.is_available()):
            ToolboxRegistry.warning(f'{self.cfg.device} is not available, using cpu instead!')
            self.cfg.device = 'cpu'

        self.torch_generator = torch.Generator(device=self.cfg.device)
        self.torch_generator.manual_seed(0)

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

    def _forward_batch(self, inputs):
        if len(inputs) > self.cfg.batch_size:
            actions = []
            for i in range(0, len(inputs), self.cfg.batch_size):
                tensor_obs = torch.tensor(inputs[i:i + self.cfg.batch_size], dtype=torch.long, device=self.cfg.device)
                batch_actions = torch.squeeze(self.net.act(tensor_obs, generator=self.torch_generator)).tolist()
                if not isinstance(batch_actions, list):
                    batch_actions = [batch_actions]
                actions.extend(batch_actions)
        else:
            tensor_obs = torch.tensor(inputs, dtype=torch.long, device=self.cfg.device)
            actions = torch.squeeze(self.net.act(tensor_obs, generator=self.torch_generator)).tolist()
            if not isinstance(actions, list):
                actions = [actions]
        return actions

    def _prepare_inputs(self, pos, observations):
        if isinstance(observations[0], dict):
            agent_positions = [obs["global_xy"] for obs in observations]
            goals = [obs["global_target_xy"] for obs in observations]

            if pos not in self._obs_generators:
                gen = ObservationGenerator(
                    observations[0]["global_obstacles"].copy().astype(int).tolist(),
                    self.input_parameters,
                )
                gen.create_agents(agent_positions, goals)
                self._obs_generators[pos] = gen
                self._last_actions[pos] = [-1] * len(observations)

            self._obs_generators[pos].update_agents(
                agent_positions, goals, self._last_actions[pos]
            )
            return self._obs_generators[pos].generate_observations()
        return observations

    def act(self, observations):
        return self.act_batch([observations])[0]

    def act_batch(self, observations_list, positions=None):
        if positions is None:
            positions = list(range(len(observations_list)))

        all_inputs = []
        env_agent_counts = []
        for pos, observations in zip(positions, observations_list):
            inputs = self._prepare_inputs(pos, observations)
            all_inputs.extend(inputs)
            env_agent_counts.append(len(inputs))

        all_actions = self._forward_batch(all_inputs)

        results = []
        offset = 0
        for pos, count in zip(positions, env_agent_counts):
            env_actions = all_actions[offset:offset + count]
            self._last_actions[pos] = env_actions.copy()
            results.append(env_actions)
            offset += count

        return results

    def reset_states(self):
        self._obs_generators = {}
        self._last_actions = {}
        self.torch_generator.manual_seed(0)