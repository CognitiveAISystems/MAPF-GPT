import ctypes
import numpy as np
from typing import Literal
from pydantic import Extra
from pogema_toolbox.algorithm_config import AlgoBase

from pogema import GridConfig

import os
import subprocess
if not os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'liblacam.so')):
    calling_script_dir = os.path.dirname(os.path.abspath(__file__))
    cmake_cmd = ['cmake', '.']
    subprocess.run(cmake_cmd, check=True, cwd=calling_script_dir)
    make_cmd = ['make', '-j8']
    subprocess.run(make_cmd, check=True, cwd=calling_script_dir)
    
    
class LacamLib:
    def __init__(self, lib_path):
        self.lib_path = lib_path
        self.load_library()
            
    def load_library(self):
        self._lacam_lib = ctypes.CDLL(self.lib_path)

        self._lacam_lib.run_lacam.argtypes = [
            ctypes.c_char_p,  # map_name
            ctypes.c_char_p,  # scene_name
            ctypes.c_int,     # N
            ctypes.c_float    # time_limit_sec
        ]
        self._lacam_lib.run_lacam.restype = ctypes.c_char_p
    
    def run_lacam(self, map_file_content, scene_file_content, num_agents, lacam_timeouts):
        map_file_bytes = map_file_content.encode('utf-8')
        scenario_file_bytes = scene_file_content.encode('utf-8')

        num_agents_int = ctypes.c_int(num_agents)
        for time_limit_sec in lacam_timeouts:
            result = self._lacam_lib.run_lacam(
                map_file_bytes, 
                scenario_file_bytes, 
                num_agents_int,
                time_limit_sec
            )

            try:
                result_str = result.decode('utf-8')
            except Exception as e:
                print(f'Exception occured while running Lacam: {e}')
                raise e
            
            if "ERROR" in result_str:
                print(f'Lacam failed to find path with time_limit_sec={time_limit_sec} | {result_str}')
            else:
                return True, result_str
        
        return False, None

class LacamAgent:
    def __init__(self, idx):
        self._moves = GridConfig().MOVES
        self._reverse_actions = {tuple(self._moves[i]): i for i in range(len(self._moves))}

        self.idx = idx
        self.previous_goal = None
        self.path = []

    def is_new_goal(self, new_goal):
        return not self.previous_goal == new_goal
    
    def set_new_goal(self, new_goal):
        self.previous_goal = new_goal

    def set_path(self, new_path):
        self.path = new_path[::-1]

    def format_task_string(self, start_xy, target_xy, map_shape):
        task_file_content = f"{self.idx}	tmp.map	{map_shape[0]}	{map_shape[1]}	"
        task_file_content += f"{start_xy[1]}	{start_xy[0]}	{target_xy[1]}	{target_xy[0]}	1\n"
        return task_file_content
    
    def get_action(self):
        action = 0
        if len(self.path) > 1:
           x, y = self.path[-1]
           tx, ty = self.path[-2]
           action = self._reverse_actions[tx - x, ty - y]
           self.path.pop()
        return action

    def clear_state(self):
        self.previous_goal = None
        self.path = []
        

class LacamInferenceConfig(AlgoBase, extra=Extra.forbid):
    name: Literal['LaCAM'] = 'LaCAM'
    time_limit: float = 60
    timeouts: list = [1.0, 5.0, 10.0, 60.0]
    lacam_lib_path: str = "lacam/liblacam.so"


class LacamInference:
    def __init__(self, cfg: LacamInferenceConfig):
        self.cfg = cfg
        self.lacam_agents = None
        self.lacam_lib = LacamLib(cfg.lacam_lib_path)
        
    def _parse_data(self, data):
        if data is None:
            return None
        lines = data.strip().split('\n')
        columns = None

        for line in lines:
            tuples = [tuple(map(int, item.split(','))) for item in line.strip().split('|') if item]
            if len(tuples) == 0:
                return None
            if columns is None:
                columns = [[] for _ in range(len(tuples))]
            for i, t in enumerate(tuples):
                columns[i].append(t[::-1])

        return columns

    def _find_near_goal(self, start_xy, target_xy, map_array, processed_targets):
        for radius in range(1, 3):
            offset_list = []
            for x_offset in range(-radius, radius+1):
                for y_offset in range(-radius, radius+1):
                    if x_offset == 0 and y_offset == 0:
                        continue
                    offset_list.append((x_offset, y_offset))

            offset_list.sort(key=lambda xy_off: (abs(xy_off[0]) + abs(xy_off[1]))**0.5)

            for (x_offset, y_offset) in offset_list:
                near_target_x = target_xy[0] + x_offset
                near_target_y = target_xy[1] + y_offset
                is_obstacle = map_array[near_target_x, near_target_y]
                assert map_array[target_xy[0], target_xy[1]] == 0
                if not is_obstacle and (near_target_x, near_target_y) not in processed_targets \
                    and start_xy != (near_target_x, near_target_y):
                    return (near_target_x, near_target_y)

    def act(self, observations, rewards=None, dones=None, info=None, skip_agents=None):
        map_array = np.array(observations[0]['global_obstacles'])
        agent_starts_xy = [obs['global_xy'] for obs in observations]
        agent_targets_xy = [obs['global_target_xy'] for obs in observations]

        has_new_tasks = False

        processed_starts = set()
        processed_targets = set()
        if self.lacam_agents is None:
            self.lacam_agents = [LacamAgent(idx) for idx in range(len(observations))]
        # Process old tasks
        agent_tasks_dict = {}
        for idx, (start_xy, target_xy) in enumerate(zip(agent_starts_xy, agent_targets_xy)):
            if self.lacam_agents[idx].is_new_goal(target_xy):
                continue
            if start_xy == target_xy or target_xy in processed_targets:
                near_target_xy = self._find_near_goal(start_xy, target_xy, map_array, processed_targets)
                target_xy = near_target_xy
                
            processed_starts.add(start_xy)
            processed_targets.add(target_xy)
                
            agent_task = self.lacam_agents[idx].format_task_string(start_xy, target_xy, map_shape=map_array.shape)
            agent_tasks_dict[idx] = agent_task
            
        # Process new tasks
        for idx, (start_xy, target_xy) in enumerate(zip(agent_starts_xy, agent_targets_xy)):
            if not self.lacam_agents[idx].is_new_goal(target_xy):
                continue
            if target_xy in processed_targets:
                near_target_xy = self._find_near_goal(start_xy, target_xy, map_array, processed_targets)
                target_xy = near_target_xy
            self.lacam_agents[idx].set_new_goal(target_xy)
            has_new_tasks = True
                
            processed_starts.add(start_xy)
            processed_targets.add(target_xy)
                
            agent_task = self.lacam_agents[idx].format_task_string(start_xy, target_xy, map_shape=map_array.shape)
            agent_tasks_dict[idx] = agent_task
            
        task_file_content = "version 1\n"
        for idx in range(len(self.lacam_agents)):
            task_file_content += agent_tasks_dict[idx]

        if has_new_tasks:
            map_row = lambda row: ''.join('@' if x else '.' for x in row)
            map_content = '\n'.join(map_row(row) for row in map_array)
            map_file_content = f"type octile\nheight {map_array.shape[0]}\nwidth {map_array.shape[1]}\nmap\n{map_content}"
            solved, lacam_results = self.lacam_lib.run_lacam(map_file_content, task_file_content, len(self.lacam_agents), self.cfg.timeouts)
            if solved:
                agent_paths = self._parse_data(lacam_results)
            else:
                agent_paths = [[agent_starts_xy[i] for _ in range(256)] for i in range(len(agent_starts_xy))] # if failed - agents just wait in start locations
            if agent_paths is not None:
                for idx, agent_path in enumerate(agent_paths):
                    self.lacam_agents[idx].set_path(agent_path)

        return [agent.get_action() for agent in self.lacam_agents]

    def after_step(self, dones):
        pass

    def reset_states(self):
        self.lacam_agents = None

    def after_reset(self):
        pass

    def get_additional_info(self):
        addinfo = {"rl_used": 0.0}
        return addinfo