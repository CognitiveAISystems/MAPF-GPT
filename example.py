import argparse
from pathlib import Path

import torch
import yaml
from pogema_toolbox.create_env import Environment
from pogema_toolbox.evaluator import run_episode
from pogema_toolbox.registry import ToolboxRegistry

from create_env import create_eval_env
from gpt.inference import MAPFGPTInference, MAPFGPTInferenceConfig


def main():
    parser = argparse.ArgumentParser(description='MAPF-GPT Inference Script')
    parser.add_argument('--animation', action='store_false', help='Enable animation (default: %(default)s)')
    parser.add_argument('--num_agents', type=int, default=32, help='Number of agents (default: %(default)d)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed (default: %(default)d)')
    parser.add_argument('--map_name', type=str, default='validation-random-seed-001', help='Map name (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use: cuda, cpu, mps (default: %(default)s)')
    parser.add_argument('--max_episode_steps', type=int, default=128,
                        help='Maximum episode steps (default: %(default)d)')
    parser.add_argument('--show_map_names', action='store_true', help='Shows names of all available maps')

    parser.add_argument('--model', type=str, choices=['2M', '6M', '85M'], default='2M',
                        help='Model to use: 2M, 6M, 85M (default: %(default)s)')

    # loading maps from eval folders
    for maps_file in Path("eval_configs").rglob('maps.yaml'):
        with open(maps_file, 'r') as f:
            maps = yaml.safe_load(f)
        ToolboxRegistry.register_maps(maps)

    args = parser.parse_args()

    if args.show_map_names:
        for map_ in ToolboxRegistry.get_maps():
            print(map_)
        return

    env_cfg = Environment(
        with_animation=args.animation,
        observation_type="MAPF",
        on_target="nothing",
        map_name=args.map_name,
        max_episode_steps=args.max_episode_steps,
        num_agents=args.num_agents,
        seed=args.seed,
        obs_radius=5,
        collision_system="soft",
    )

    # pytorch seeding
    torch_seed = 42
    torch.manual_seed(torch_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(torch_seed)
    torch.backends.mps.is_available()
    torch.backends.cudnn.deterministic = True

    env = create_eval_env(env_cfg)
    algo = MAPFGPTInference(MAPFGPTInferenceConfig(path_to_weights=f'weights/model-{args.model}.pt', device=args.device))
    algo.reset_states()
    results = run_episode(env, algo)

    svg_path = f"svg/{args.map_name}-{args.model}-seed-{args.seed}.svg"
    env.save_animation(svg_path)
    ToolboxRegistry.info(f'Saved animation to: {svg_path}')

    ToolboxRegistry.success(results)


if __name__ == "__main__":
    main()
