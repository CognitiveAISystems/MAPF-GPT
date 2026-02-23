import numpy as np
import cppimport.import_hook
from tokenizer import cost2go
from tokenizer.encoder import AgentsInfo, Encoder, InputParameters
from tokenizer.parameters import InputParameters as PyInputParameters


class ObservationGenerator:
    def __init__(self, maps, data, cfg: PyInputParameters):
        self.moves = [[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]]
        self.moves_dict = {
            (0, 0): "w",
            (-1, 0): "u",
            (1, 0): "d",
            (0, -1): "l",
            (0, 1): "r",
        }
        self.cfg = cfg
        self.maps = maps
        self.data = data
        cpp_cfg = InputParameters(
            cfg.cost2go_value_limit,
            cfg.num_agents,
            cfg.num_previous_actions,
            cfg.context_size,
        )
        self.encoder = Encoder(cpp_cfg)
        self.inputs = []
        self.gt_actions = []
        self.cost2go_data = None

    def generate_observations(self, start_range, end_range):
        def find_last_non_zero_index(actions):
            for i in reversed(range(len(actions))):
                if actions[i] != 0:
                    return i
            return len(actions)

        self.inputs = []
        self.gt_actions = []
        map_name = ""

        for instance_id in range(start_range, end_range):
            if self.data[instance_id]["metrics"].get("CSR", 1) < 1:
                continue
            if map_name != self.data[instance_id]["env_grid_search"]["map_name"]:
                grid = self.get_grid_map(
                    self.data[instance_id]["env_grid_search"]["map_name"]
                )
                self.cost2go_data = cost2go.precompute_cost2go(
                    grid, self.cfg.cost2go_radius
                )
                map_name = self.data[instance_id]["env_grid_search"]["map_name"]
            paths = self.get_agent_paths(
                self.data[instance_id]["metrics"]["made_actions"],
                self.data[instance_id]["metrics"]["init_positions"],
            )
            goal_positions = None
            if "global_lifelong_targets_xy" in self.data[instance_id]["metrics"].keys():
                goal_positions = self.get_goal_positions(
                    paths,
                    self.data[instance_id]["metrics"]["global_lifelong_targets_xy"],
                )
            proximity_lists = self.generate_agent_proximity(paths)
            for agent_id in range(len(paths)):
                self.data[instance_id]["metrics"]["made_actions"][agent_id].append(0)

            for agent_id in range(len(paths)):
                goal_t = find_last_non_zero_index(
                    self.data[instance_id]["metrics"]["made_actions"][agent_id]
                )
                for t in range(len(paths[agent_id])):
                    agents = self.generate_agent_info(
                        agent_id, t, proximity_lists[t][agent_id], paths, goal_positions
                    )
                    cost2go_obs = self.generate_cost2go_obs(
                        self.cost2go_data[tuple(paths[agent_id][-1])],
                        tuple(paths[agent_id][t]),
                    )

                    self.inputs.append(
                        np.array(
                            self.encoder.encode(agents, cost2go_obs), dtype=np.int8
                        )
                    )
                    self.gt_actions.append(
                        self.data[instance_id]["metrics"]["made_actions"][agent_id][t]
                    )
                    if t > goal_t:
                        self.gt_actions[-1] = 5  # 5 is a wait in goal

        return self.inputs, self.gt_actions

    def str_map_to_list(self, str_map, free, obstacle):
        obstacles = []
        for row_idx, line in enumerate(str_map.split()):
            row = []
            for char in line:
                if char == ".":
                    row.append(free)
                elif char == "#":
                    row.append(obstacle)
                else:
                    raise KeyError(f"Unsupported symbol '{char}' at line {row_idx}")
            if row:
                assert (
                    len(obstacles[-1]) == len(row) if obstacles else True
                ), f"Wrong string size for row {row_idx};"
                obstacles.append(row)

        return obstacles

    def get_grid_map(self, map_name):
        grid_map = self.str_map_to_list(self.maps[map_name], 0, 1)
        grid_map = np.array(grid_map)
        rows, cols = grid_map.shape
        border_size = self.cfg.cost2go_radius
        new_grid = np.ones(
            (rows + 2 * border_size, cols + 2 * border_size), dtype=grid_map.dtype
        )
        new_grid[border_size : border_size + rows, border_size : border_size + cols] = (
            grid_map
        )

        return new_grid

    def generate_agent_proximity(self, paths):
        num_agents = len(paths)
        num_timesteps = len(paths[0])
        proximity_lists = [[] for _ in range(num_timesteps)]

        for t in range(num_timesteps):
            coordinates = [paths[agent_id][t] for agent_id in range(num_agents)]
            proximity_at_t = [[] for _ in range(num_agents)]

            for i in range(num_agents):
                distances = []
                for j in range(num_agents):
                    dx = abs(coordinates[i][0] - coordinates[j][0])
                    dy = abs(coordinates[i][1] - coordinates[j][1])
                    if dx <= self.cfg.agents_radius and dy <= self.cfg.agents_radius:
                        distance = self.cost2go_data[tuple(coordinates[i])][
                            coordinates[j][0]
                        ][coordinates[j][1]]
                        if distance >= 0:
                            distances.append((j, distance))

                distances.sort(key=lambda x: (x[1], x[0]))
                sorted_agents = [agent_id for agent_id, _ in distances]
                proximity_at_t[i] = sorted_agents

            proximity_lists[t] = proximity_at_t

        return proximity_lists

    def get_goal_positions(self, paths, global_lifelong_targets_xy):
        all_goal_positions = []
        for i, path in enumerate(paths):
            goal_positions = []
            cur_goal_id = 0
            for pos in path:
                if pos == global_lifelong_targets_xy[i][cur_goal_id]:
                    cur_goal_id += 1
                goal_positions.append(global_lifelong_targets_xy[i][cur_goal_id])
            all_goal_positions.append(goal_positions)
        return all_goal_positions

    def get_agent_paths(self, actions_list, initial_positions):
        if self.cfg.cost2go_radius != 5:
            for i in range(len(initial_positions)):
                initial_positions[i] = [
                    initial_positions[i][0] + self.cfg.cost2go_radius - 5,
                    initial_positions[i][1] + self.cfg.cost2go_radius - 5,
                ]
        all_paths = []
        for initial_position, actions in zip(initial_positions, actions_list):
            path = [initial_position.copy()]
            current_position = initial_position.copy()
            for action in actions:
                move = self.moves[action]
                current_position[0] += move[0]
                current_position[1] += move[1]
                path.append(current_position.copy())
            all_paths.append(path)

        return all_paths

    def get_agent_info(self, agent_id, time, paths, goal_positions, central_position):
        path = paths[agent_id]
        agent_position = tuple(path[min(time, len(path) - 1)])
        if goal_positions is not None:
            goal_position = tuple(goal_positions[agent_id][min(time, len(path) - 1)])
        else:
            goal_position = tuple(path[-1])
        relative_coords = (
            agent_position[0] - central_position[0],
            agent_position[1] - central_position[1],
        )
        relative_goal = (
            goal_position[0] - central_position[0],
            goal_position[1] - central_position[1],
        )

        if time < self.cfg.num_previous_actions:
            previous_actions = ["n"] * (self.cfg.num_previous_actions - time) + [
                self.moves_dict[
                    (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
                ]
                for i in range(1, min(time + 1, len(path) - 1))
            ]
        else:
            previous_actions = [
                self.moves_dict[
                    (path[i][0] - path[i - 1][0], path[i][1] - path[i - 1][1])
                ]
                for i in range(
                    time - self.cfg.num_previous_actions + 1,
                    min(time + 1, len(path) - 1),
                )
            ]
        if len(previous_actions) < self.cfg.num_previous_actions:
            previous_actions += ["w"] * (
                self.cfg.num_previous_actions - len(previous_actions)
            )

        next_action = ""
        for m in self.moves[1:]:
            new_pos = (agent_position[0] + m[0], agent_position[1] + m[1])
            if (
                self.cost2go_data[goal_position][new_pos[0]][new_pos[1]] >= 0
                and self.cost2go_data[goal_position][agent_position[0]][
                    agent_position[1]
                ]
                > self.cost2go_data[goal_position][new_pos[0]][new_pos[1]]
            ):
                next_action += "1"
            else:
                next_action += "0"

        return AgentsInfo(relative_coords, relative_goal, previous_actions, next_action)

    def generate_agent_info(
        self, agent_id, time, proximity_list, paths, goal_positions
    ):
        agents_info = []
        for id in proximity_list[: self.cfg.num_agents]:
            agent_info = self.get_agent_info(
                id, time, paths, goal_positions, paths[agent_id][time]
            )
            agents_info.append(agent_info)

        return agents_info

    def generate_cost2go_obs(self, cost2go_data, pos):
        observation = cost2go.generate_cost2go_obs(
            cost2go_data,
            pos,
            self.cfg.cost2go_radius,
            self.cfg.cost2go_value_limit,
            self.cfg.mask_cost2go,
        )
        return observation
