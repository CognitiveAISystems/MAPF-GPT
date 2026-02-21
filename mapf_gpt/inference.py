# from pathlib import Path
# from typing import Literal, Optional
#
# import cppimport.import_hook
# import torch
# from huggingface_hub import hf_hub_download
# from pogema_toolbox.algorithm_config import AlgoBase
# from pogema_toolbox.registry import ToolboxRegistry
# from pydantic import Extra
#
# from mapf_gpt.model import GPT, GPTConfig
# from tokenizer import cost2go
# from tokenizer.tokenizer import Encoder, InputParameters

from pathlib import Path
from typing import Literal, Optional

import cppimport.import_hook
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
        self.observation_generator = None
        self.last_actions = None

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

    def act(self, observations):
        if isinstance(observations[0], dict):
            positions = [obs["global_xy"] for obs in observations]
            goals = [obs["global_target_xy"] for obs in observations]
            if self.observation_generator is None:
                self.observation_generator = ObservationGenerator(observations[0]["global_obstacles"].copy().astype(int).tolist(), self.input_parameters)
                self.observation_generator.create_agents(positions, goals)
                self.last_actions = [-1 for _ in range(len(observations))]
            self.observation_generator.update_agents(positions, goals, self.last_actions)
            inputs = self.observation_generator.generate_observations()
        else:
            inputs = observations
        if len(inputs) > self.cfg.batch_size:
            actions = []
            for i in range(0, len(inputs), self.cfg.batch_size):
                batch_inputs = inputs[i:i + self.cfg.batch_size]
                tensor_obs = torch.tensor(batch_inputs, dtype=torch.long, device=self.cfg.device)
                batch_actions = torch.squeeze(self.net.act(tensor_obs, generator=self.torch_generator)).tolist()
                actions.extend(batch_actions)
        else:
            tensor_obs = torch.tensor(inputs, dtype=torch.long, device=self.cfg.device)
            actions = torch.squeeze(self.net.act(tensor_obs, generator=self.torch_generator)).tolist()
        if not isinstance(actions, list):
            actions = [actions]
        self.last_actions = actions.copy()
        return actions

    def reset_states(self):
        self.observation_generator = None
        self.torch_generator.manual_seed(0)