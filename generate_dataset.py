import glob
import os
import json
import yaml
import hashlib
import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
import shutil
import multiprocessing as mp
from functools import partial
from pathlib import Path

from pogema_toolbox.create_env import Environment
from pogema_toolbox.eval_utils import initialize_wandb, save_evaluation_results
from pogema_toolbox.evaluator import evaluation
from pogema_toolbox.registry import ToolboxRegistry

from create_env import create_logging_env
from lacam.inference import LacamInference, LacamInferenceConfig
from tokenizer.generate_observations import ObservationGenerator
from tokenizer.parameters import InputParameters

EXPERT_DATA_FOLDER = "LaCAM_data"
TEMP_FOLDER = "temp"
DATASET_FOLDER = "dataset"
CONFIGS = [
    "dataset_configs/10-medium-mazes/10-medium-mazes-part1.yaml",
    "dataset_configs/10-medium-mazes/10-medium-mazes-part2.yaml",
    "dataset_configs/10-medium-mazes/10-medium-mazes-part3.yaml",
    "dataset_configs/10-medium-mazes/10-medium-mazes-part4.yaml",
    "dataset_configs/12-medium-random/12-medium-random-part1.yaml",
]

RANDOM_MAPS_FOLDER = "dataset_configs/12-medium-random"
MAZES_MAPS_FOLDER = "dataset_configs/10-medium-mazes"

NUM_CHUNKS = 50
FILE_PER_CHUNK = 10
DESIRED_SIZE = 10*2**21 # per chunk
MAZE_RATIO = 0.9
NUM_PROCESSES = 50

def tensor_to_hash(tensor):
    tensor_bytes = tensor.tobytes()
    return hashlib.sha256(tensor_bytes).hexdigest()

def get_files_by_type(folder_path):
    all_files = glob.glob(os.path.join(folder_path, '*.json'))
    maze_files = [f for f in all_files if 'mazes' in os.path.basename(f).lower()]
    random_files = [f for f in all_files if 'random' in os.path.basename(f).lower()]
    maze_files.sort()
    random_files.sort()

    return maze_files, random_files

def generate_part(map_name, maps):
    print("processing map", map_name)
    cfg = InputParameters()
    with open(map_name, "r") as f:
        data = json.load(f)
    generator = ObservationGenerator(maps, data, cfg)
    tensors, gt_actions = generator.generate_observations(0, len(data))
    return tensors, gt_actions

def balance_and_filter_tensors(tensors, actions, known_hashes=None):
    new_tensors = []
    new_actions = []
    duplicates = 0
    if known_hashes is None:
        known_hashes = set()
    for tensor, action in zip(tensors, actions):
        tensor_hash = tensor_to_hash(tensor)

        if tensor_hash not in known_hashes:
            known_hashes.add(tensor_hash)
            new_tensors.append(tensor)
            new_actions.append(action)
        else:
            duplicates += 1
    if len(new_tensors) > 0:
        actions_made = [0 for i in range(6)]
        for action in new_actions:
            actions_made[action] += 1
        i = len(new_tensors) - 1
        discarded = 0
        while i >= 0:
            if new_actions[i] == 5:
                if (actions_made[0] + actions_made[5]) > len(new_tensors) // 5:
                    new_actions.pop(i)
                    new_tensors.pop(i)
                    actions_made[5] -= 1
                    discarded += 1
                else:
                    new_actions[i] = 0
            i -= 1
        print(discarded, duplicates, len(new_tensors), actions_made)
        new_tensors = np.array(new_tensors)
        new_actions = np.array(new_actions)
        indices = np.arange(len(new_tensors))
        np.random.shuffle(indices) # shuffle to balance actions
        new_tensors = new_tensors[indices]
        new_actions = new_actions[indices]
    return new_tensors, new_actions

def calculate_elements_to_pick(data, total_pick_count):
    file_elements = {}
    total_elements = 0
    for file, (tensors, actions) in data.items():
        total_elements += len(tensors)
        file_elements[file] = len(tensors)
    if total_pick_count > total_elements:
        print(
            f"Warning! Files don't contain enough data to pick {total_pick_count} elements. Using {total_elements} elements instead"
        )
        total_pick_count = total_elements

    elements_to_pick = {}
    total_picked = 0
    for file_path, num_elements in file_elements.items():
        elements_to_pick[file_path] = int(
            num_elements * total_pick_count / total_elements
        )
        total_picked += elements_to_pick[file_path]

    while total_picked < total_pick_count:
        for file_path, num_elements in file_elements.items():
            if total_picked == total_pick_count:
                break
            if elements_to_pick[file_path] < num_elements:
                elements_to_pick[file_path] += 1
                total_picked += 1

    return elements_to_pick, total_pick_count

def process_file(file, maps):
    tensors, actions = generate_part(file, maps)
    tensors, actions = balance_and_filter_tensors(tensors, actions)
    return file, tensors, actions

def process_files(maze_files, random_files, output_file):
    maps_random = yaml.safe_load(open(f"{RANDOM_MAPS_FOLDER}/maps.yaml", "r"))
    maps_mazes = yaml.safe_load(open(f"{MAZES_MAPS_FOLDER}/maps.yaml", "r"))
    maze_desired_size = int(DESIRED_SIZE * MAZE_RATIO)
    random_desired_size = DESIRED_SIZE - maze_desired_size
    
    # Process maze files
    with mp.Pool(NUM_PROCESSES) as pool:
        maze_results = pool.map(partial(process_file, maps=maps_mazes), maze_files)
    
    # Process random files
    with mp.Pool(NUM_PROCESSES) as pool:
        random_results = pool.map(partial(process_file, maps=maps_random), random_files)
    
    # Combine results into dictionaries
    maze_data = {file: (tensors, actions) for file, tensors, actions in maze_results}
    random_data = {file: (tensors, actions) for file, tensors, actions in random_results}

    
    # Pick required portion from each file
    maze_elements_to_pick, total_maze_elements = calculate_elements_to_pick(maze_data, maze_desired_size)
    random_elements_to_pick, total_random_elements = calculate_elements_to_pick(random_data, random_desired_size)
    
    all_tensors = np.empty((total_maze_elements + total_random_elements, 256), dtype=np.int8)
    all_actions = np.empty(total_maze_elements + total_random_elements, dtype=np.int8)
    
    current_index = 0
    for file_path, pick_count in maze_elements_to_pick.items():
        if pick_count > 0:
            all_tensors[current_index:current_index+pick_count] = maze_data[file_path][0][:pick_count]
            all_actions[current_index:current_index+pick_count] = maze_data[file_path][1][:pick_count]
        current_index += pick_count
    for file_path, pick_count in random_elements_to_pick.items():
        if pick_count > 0:
            all_tensors[current_index:current_index+pick_count] = random_data[file_path][0][:pick_count]
            all_actions[current_index:current_index+pick_count] = random_data[file_path][1][:pick_count]
        current_index += pick_count
    
    # Shuffle the data
    indices = np.arange(len(all_tensors))
    np.random.shuffle(indices)
    all_tensors = all_tensors[indices]
    all_actions = all_actions[indices]
    
    # Save the data
    schema = pa.schema([
        ('input_tensors', pa.list_(pa.int8())),
        ('gt_actions', pa.int8())
    ])
    num_samples = len(all_tensors)
    samples_per_file = num_samples // FILE_PER_CHUNK
    
    for i in range(FILE_PER_CHUNK):
        start_idx = i * samples_per_file
        end_idx = start_idx + samples_per_file if i < FILE_PER_CHUNK - 1 else num_samples
        
        tensors_chunk = all_tensors[start_idx:end_idx]
        actions_chunk = all_actions[start_idx:end_idx]
        
        input_tensors_col = pa.array(tensors_chunk.tolist(), type=pa.list_(pa.int8()))
        gt_actions_col = pa.array(actions_chunk)
        
        table = pa.Table.from_arrays([input_tensors_col, gt_actions_col], schema=schema)
        
        chunk_output_file = f"{output_file}_part_{i}.arrow"
        with open(chunk_output_file, "wb") as f:
            with ipc.new_file(f, schema) as writer:
                writer.write(table)
        
        print(f"Saved {chunk_output_file} with {len(tensors_chunk)} samples")

def run_expert():
    env_cfg_name = "Environment"
    ToolboxRegistry.register_env(env_cfg_name, create_logging_env, Environment)
    ToolboxRegistry.register_algorithm("LaCAM", LacamInference, LacamInferenceConfig)
    unique_paths = {os.path.dirname(path) for path in CONFIGS}
    maps = {}
    for path in unique_paths:
        with open(f"{path}/maps.yaml", "r") as f:
            folder_maps = yaml.safe_load(f)
            maps.update(folder_maps)
    ToolboxRegistry.register_maps(maps)
    for config in CONFIGS:
        with open(config, "r") as f:
            evaluation_config = yaml.safe_load(f)

        eval_dir = Path(EXPERT_DATA_FOLDER) / config[:-5]
        initialize_wandb(evaluation_config, eval_dir, False, EXPERT_DATA_FOLDER)
        evaluation(evaluation_config, eval_dir=eval_dir)
        save_evaluation_results(eval_dir)

def split_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    data_per_map = {}
    for d in data:
        if d["env_grid_search"]["map_name"] not in data_per_map:
            data_per_map[d["env_grid_search"]["map_name"]] = []
        data_per_map[d["env_grid_search"]["map_name"]].append(d)

    os.makedirs(TEMP_FOLDER, exist_ok=True)
    for k, v in data_per_map.items():
        with open(f"{TEMP_FOLDER}/{k}.json", "w") as f:
            json.dump(v, f)

def generate_chunks():
    maze_files, random_files = get_files_by_type(TEMP_FOLDER)
    
    maze_chunk_size = len(maze_files) // NUM_CHUNKS
    random_chunk_size = len(random_files) // NUM_CHUNKS
    
    maze_chunks = [maze_files[i:i+maze_chunk_size] for i in range(0, len(maze_files), maze_chunk_size)]
    random_chunks = [random_files[i:i+random_chunk_size] for i in range(0, len(random_files), random_chunk_size)]
    
    for i in range(NUM_CHUNKS):
        process_files(maze_chunks[i], random_chunks[i], f"{DATASET_FOLDER}/chunk_{i}")
        
def main():
    # Step 1: Run LaCAM to obtain expert data in json format.
    run_expert()

    # Step 2: Load one (or mutiple) big json file and split it (them) into small ones (1 map = 1 json).
    files = [f"{EXPERT_DATA_FOLDER}/{config[:-5]}/LaCAM.json" for config in CONFIGS]
    with mp.Pool() as pool:
        pool.map(split_json, files)
    
    # Step 3: Generate dataset with chunk files.
    generate_chunks()
    
    #Step 4: clear temp folder
    shutil.rmtree(TEMP_FOLDER)

if __name__ == "__main__":
    main()
