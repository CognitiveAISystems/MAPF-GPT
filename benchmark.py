from pathlib import Path

import yaml
from pogema_toolbox.create_env import Environment
from pogema_toolbox.eval_utils import initialize_wandb, save_evaluation_results
from pogema_toolbox.evaluator import evaluation
from pogema_toolbox.registry import ToolboxRegistry

from create_env import create_eval_env
from gpt.inference import MAPFGPTInference, MAPFGPTInferenceConfig

PROJECT_NAME = "Benchmark"
BASE_PATH = Path("eval_configs")


def ensure_weights(eval_config):
    for algo_name, algo_cfg in eval_config['algorithms'].items():
        ToolboxRegistry.create_algorithm(algo_cfg['name'], **algo_cfg)


def main(disable_wandb=False):
    env_cfg_name = "Environment"
    ToolboxRegistry.register_env(env_cfg_name, create_eval_env, Environment)
    ToolboxRegistry.register_algorithm(
        "MAPF-GPT", MAPFGPTInference, MAPFGPTInferenceConfig
    )

    folder_names = [
        "01-random",
        "02-mazes",
        "03-warehouse",
        "04-movingai",
        "05-puzzles",
    ]

    for folder in folder_names:
        maps_path = BASE_PATH / folder / "maps.yaml"
        with open(maps_path, "r") as f:
            maps = yaml.safe_load(f)
        ToolboxRegistry.register_maps(maps)

        config_path = BASE_PATH / folder / f"{Path(folder).name}.yaml"
        with open(config_path) as f:
            evaluation_config = yaml.safe_load(f)

        # ensuring model weights are downloaded
        ensure_weights(evaluation_config)

        eval_dir = BASE_PATH / folder
        initialize_wandb(evaluation_config, eval_dir, disable_wandb, PROJECT_NAME)
        evaluation(evaluation_config, eval_dir=eval_dir)
        save_evaluation_results(eval_dir)


if __name__ == "__main__":
    main()
