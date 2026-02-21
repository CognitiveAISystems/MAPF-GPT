// cppimport
#include "observation_generator.h"

void ObservationGenerator::mark_components()
{
    components = std::vector<std::vector<int>>(grid.size(), std::vector<int>(grid[0].size(), 0));
    std::vector<std::pair<int, int>> moves = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    int current_component = 1;

    for (size_t i = 0; i < grid.size(); i++)
    {
        for (size_t j = 0; j < grid[0].size(); j++)
        {
            if (grid[i][j] != 0 || components[i][j] != 0)
                continue;
            std::queue<std::pair<int, int>> fringe;
            fringe.push({i, j});
            components[i][j] = current_component;

            while (!fringe.empty())
            {
                auto [ci, cj] = fringe.front();
                fringe.pop();
                for (const auto &move : moves)
                {
                    int ni = ci + move.first;
                    int nj = cj + move.second;
                    if (ni >= 0 && ni < grid.size() &&
                        nj >= 0 && nj < grid[0].size() &&
                        grid[ni][nj] == 0 &&
                        components[ni][nj] == 0)
                    {
                        components[ni][nj] = current_component;
                        fringe.push({ni, nj});
                    }
                }
            }
            current_component++;
        }
    }
}

void ObservationGenerator::precompute_cost2go()
{
    std::vector<std::pair<int, int>> precomputed_cells;
    for (int i = 0; i < grid.size(); i += cfg.grid_step)
        for (int j = 0; j < grid[0].size(); j++)
            if (grid[i][j] == 0)
                precomputed_cells.push_back({i, j});
    for (int i = 0; i < grid.size(); i++)
        for (int j = 0; j < grid[0].size(); j += cfg.grid_step)
            if (grid[i][j] == 0)
                precomputed_cells.push_back({i, j});
    int current_idx = 0;
    for (const auto &cell : precomputed_cells)
        if (precomputed_cells_map.find(cell) == precomputed_cells_map.end())
        {
            precomputed_cells_map[cell] = current_idx;
            current_idx++;
        }

    if (cfg.save_cost2go)
    {
        std::ifstream ifile("precomputed_cost2go.bin", std::ios::binary);
        if (ifile.is_open())
        {
            size_t rows, cols;
            ifile.read(reinterpret_cast<char *>(&rows), sizeof(size_t));
            ifile.read(reinterpret_cast<char *>(&cols), sizeof(size_t));

            precomputed_cost2go.resize(rows, std::vector<uint16_t>(cols));
            for (size_t i = 0; i < rows; i++)
            {
                ifile.read(reinterpret_cast<char *>(precomputed_cost2go[i].data()), cols * sizeof(uint16_t));
            }

            ifile.close();
            return;
        }
    }

    precomputed_cost2go = std::vector<std::vector<uint16_t>>(precomputed_cells_map.size(), std::vector<uint16_t>(precomputed_cells_map.size(), std::numeric_limits<uint16_t>::max()));
#pragma omp parallel for
    for (size_t i = 0; i < precomputed_cells.size(); i++)
    {
        const auto &cell = precomputed_cells[i];
        if (grid[cell.first][cell.second] != 0)
            continue;
        std::vector<std::pair<int, int>> moves = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
        std::queue<std::pair<int, int>> fringe;
        fringe.push(cell);
        std::vector<std::vector<uint16_t>> cost_matrix(grid.size(), std::vector<uint16_t>(grid[0].size(), std::numeric_limits<uint16_t>::max()));
        cost_matrix[cell.first][cell.second] = 0;
        while (!fringe.empty())
        {
            auto pos = fringe.front();
            fringe.pop();
            for (const auto &move : moves)
            {
                int new_i(pos.first + move.first), new_j(pos.second + move.second);
                if (new_i >= 0 && new_j >= 0 && new_i < grid.size() && new_j < grid.front().size())
                {
                    if (grid[new_i][new_j] == 0 && cost_matrix[new_i][new_j] == std::numeric_limits<uint16_t>::max())
                    {
                        cost_matrix[new_i][new_j] = cost_matrix[pos.first][pos.second] + 1;
                        fringe.push(std::make_pair(new_i, new_j));
                    }
                }
            }
        }
        for (const auto &t_cell : precomputed_cells)
            precomputed_cost2go[precomputed_cells_map[cell]][precomputed_cells_map[t_cell]] = cost_matrix[t_cell.first][t_cell.second];
    }
    if (!cfg.save_cost2go)
        return;
    std::ofstream file("precomputed_cost2go.bin", std::ios::binary);
    if (file.is_open())
    {
        size_t rows = precomputed_cost2go.size();
        size_t cols = precomputed_cost2go[0].size();

        file.write(reinterpret_cast<const char *>(&rows), sizeof(size_t));
        file.write(reinterpret_cast<const char *>(&cols), sizeof(size_t));

        for (size_t i = 0; i < rows; i++)
        {
            file.write(reinterpret_cast<const char *>(precomputed_cost2go[i].data()), cols * sizeof(uint16_t));
        }

        file.close();
    }
}

std::pair<std::vector<std::pair<int, int>>, std::vector<std::vector<uint16_t>>> ObservationGenerator::get_goal_border_and_cost2go(const std::pair<int, int> &goal)
{
    size_t left_border = goal.first / cfg.grid_step * cfg.grid_step;
    size_t right_border = std::min(left_border + cfg.grid_step, grid.size() - 1);
    size_t top_border = goal.second / cfg.grid_step * cfg.grid_step;
    size_t bottom_border = std::min(top_border + cfg.grid_step, grid[0].size() - 1);
    
    std::vector<std::pair<int, int>> goal_border;
    for (int i = left_border; i <= right_border; i++)
    {
        goal_border.push_back(std::make_pair(i, top_border));
        if (top_border + cfg.grid_step < grid[0].size())
            goal_border.push_back(std::make_pair(i, bottom_border));
    }
    for (int j = top_border; j <= bottom_border; j++)
    {
        goal_border.push_back(std::make_pair(left_border, j));
        if (left_border + cfg.grid_step < grid.size())
            goal_border.push_back(std::make_pair(right_border, j));
    }

    std::vector<std::pair<int, int>> moves = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    std::queue<std::pair<int, int>> fringe;
    fringe.push(goal);
    std::vector<std::vector<uint16_t>> cost_matrix(grid.size(), std::vector<uint16_t>(grid[0].size(), std::numeric_limits<uint16_t>::max()));
    cost_matrix[goal.first][goal.second] = 0;
    while (!fringe.empty())
    {
        auto pos = fringe.front();
        fringe.pop();
        for (const auto &move : moves)
        {
            int new_i(pos.first + move.first), new_j(pos.second + move.second);
            if (new_i >= left_border && new_j >= top_border && new_i <= right_border && new_j <= bottom_border)
                if (grid[new_i][new_j] == 0 && cost_matrix[new_i][new_j] == std::numeric_limits<uint16_t>::max())
                {
                    cost_matrix[new_i][new_j] = cost_matrix[pos.first][pos.second] + 1;
                    fringe.push(std::make_pair(new_i, new_j));
                }
        }
    }
    return std::make_pair(goal_border, cost_matrix);
}

std::vector<std::pair<int, int>> ObservationGenerator::get_cells_on_border(const std::pair<int, int> &pos)
{
    std::vector<std::pair<int, int>> cells;
    size_t left_border = std::max(pos.first - cfg.obs_radius, 0) / cfg.grid_step * cfg.grid_step;
    size_t right_border = left_border + 2 * cfg.grid_step;
    size_t top_border = std::max(pos.second - cfg.obs_radius, 0) / cfg.grid_step * cfg.grid_step;
    size_t bottom_border = top_border + 2 * cfg.grid_step;
    for (size_t i = left_border; i < std::min(right_border, grid.size()); i++)
    {
        cells.push_back(std::make_pair(i, top_border));
        if (bottom_border < grid[0].size())
            cells.push_back(std::make_pair(i, bottom_border));
    }
    for (size_t j = top_border; j < std::min(bottom_border, grid[0].size()); j++)
    {
        cells.push_back(std::make_pair(left_border, j));
        if (right_border < grid.size())
            cells.push_back(std::make_pair(right_border, j));
    }
    return cells;
}

void ObservationGenerator::compute_cost2go_partial(int agent_idx)
{
    std::vector<std::pair<int, int>> moves = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    const auto &goal = agents[agent_idx].goal;
    const auto &pos = agents[agent_idx].pos;

    auto pos_border = get_cells_on_border(pos);
    size_t left_border = std::max(pos.first - cfg.obs_radius, 0) / cfg.grid_step * cfg.grid_step;
    size_t right_border = std::min(left_border + 2 * cfg.grid_step, grid.size() - 1);
    size_t top_border = std::max(pos.second - cfg.obs_radius, 0) / cfg.grid_step * cfg.grid_step;
    size_t bottom_border = std::min(top_border + 2 * cfg.grid_step, grid[0].size() - 1);
    std::vector<std::vector<uint16_t>> cost2go(right_border - left_border + 1, std::vector<uint16_t>(bottom_border - top_border + 1, std::numeric_limits<uint16_t>::max()));
    std::vector<std::vector<uint16_t>> cost_matrix(grid.size(), std::vector<uint16_t>(grid[0].size(), std::numeric_limits<uint16_t>::max()));

    if (grid.size() <= cfg.grid_step && grid[0].size() <= cfg.grid_step)
    {
        Cost2GoPartial partial(goal, left_border, right_border, top_border, bottom_border);
        partial.cost2go = get_goal_border_and_cost2go(goal).second;
        cost2go_partials[agent_idx] = partial;
        return;
    }

    auto [goal_border, goal_cost_matrix] = get_goal_border_and_cost2go(goal);
    std::priority_queue<std::pair<int, std::pair<int, int>>, std::vector<std::pair<int, std::pair<int, int>>>, std::greater<>> pq; // pq with ascending order of cost
    for (const auto &cell : pos_border)
    {
        if (grid[cell.first][cell.second] != 0)
            continue;
        uint16_t min_cost = std::numeric_limits<uint16_t>::max();
        for (const auto &goal_cell : goal_border)
        {
            if (grid[goal_cell.first][goal_cell.second] != 0 || goal_cost_matrix[goal_cell.first][goal_cell.second] == std::numeric_limits<uint16_t>::max())
                continue;
            int new_cost = goal_cost_matrix[goal_cell.first][goal_cell.second] + precomputed_cost2go[precomputed_cells_map[std::make_pair(goal_cell.first, goal_cell.second)]][precomputed_cells_map[std::make_pair(cell.first, cell.second)]];
            if (min_cost > new_cost)
                min_cost = new_cost;
        }
        if (min_cost != std::numeric_limits<uint16_t>::max())
            pq.push({min_cost, cell});
    }
    //  if goal inside the desired cost2go square, add it to pq
    if (goal.first >= left_border && goal.first <= right_border && goal.second >= top_border && goal.second <= bottom_border)
    {
        pq.push({0, goal});
        cost_matrix[goal.first][goal.second] = 0;
    }
    std::queue<std::pair<int, int>> fringe;
    fringe.push(pq.top().second);
    cost_matrix[pq.top().second.first][pq.top().second.second] = pq.top().first;
    pq.pop();
    while (!fringe.empty())
    {
        auto current = fringe.front();
        fringe.pop();
        int current_cost = cost_matrix[current.first][current.second];
        // Check pq for elements with matching cost and add them to fringe
        while (!pq.empty() && pq.top().first == current_cost)
        {
            fringe.push(pq.top().second);
            cost_matrix[pq.top().second.first][pq.top().second.second] = current_cost;
            pq.pop();
        }
        for (const auto &move : moves)
        {
            int new_i(current.first + move.first), new_j(current.second + move.second);
            if (new_i >= left_border && new_i <= right_border &&
                new_j >= top_border && new_j <= bottom_border &&
                grid[new_i][new_j] == 0 && cost_matrix[new_i][new_j] == std::numeric_limits<uint16_t>::max())
            {
                cost_matrix[new_i][new_j] = current_cost + 1;
                fringe.push(std::make_pair(new_i, new_j));
            }
        }
        if (fringe.empty() && !pq.empty())
        {
            fringe.push(pq.top().second);
            cost_matrix[pq.top().second.first][pq.top().second.second] = pq.top().first;
            pq.pop();
        }
    }
    for (int i = left_border; i <= right_border; i++)
        std::copy(cost_matrix[i].begin() + top_border, cost_matrix[i].begin() + bottom_border + 1, cost2go[i - left_border].begin());

    Cost2GoPartial partial(goal, left_border, right_border, top_border, bottom_border);
    partial.cost2go = cost2go;
    cost2go_partials[agent_idx] = partial;
}

void ObservationGenerator::generate_cost2go_obs(int agent_idx, bool only_obstacles, std::vector<std::vector<int>> &buffer)
{
    const auto &partial = cost2go_partials[agent_idx];
    int x = agents[agent_idx].pos.first - partial.left_border - cfg.obs_radius;
    int y = agents[agent_idx].pos.second - partial.top_border - cfg.obs_radius;
    if (only_obstacles)
        for (int i = 0; i <= cfg.obs_radius * 2; i++)
            for (int j = 0; j <= cfg.obs_radius * 2; j++)
                buffer[i][j] = bool(partial.cost2go[x + i][y + j] < 0);
    int middle_value = partial.cost2go[x + cfg.obs_radius][y + cfg.obs_radius];
    for (int i = 0; i <= cfg.obs_radius * 2; i++)
        for (int j = 0; j <= cfg.obs_radius * 2; j++)
        {
            int value = partial.cost2go[x + i][y + j];
            if (value != std::numeric_limits<uint16_t>::max())
            {
                value -= middle_value;
                buffer[i][j] = value > cfg.cost2go_value_limit ? cfg.cost2go_value_limit * 2 : value < -cfg.cost2go_value_limit ? -cfg.cost2go_value_limit * 2
                                                                                                                                : value;
            }
            else
                buffer[i][j] = -cfg.cost2go_value_limit * 4;
        }
}

int ObservationGenerator::get_distance(int agent_idx, const std::pair<int, int> &pos)
{
    const auto &partial = cost2go_partials[agent_idx];
    if (pos.first < partial.left_border || pos.first >= partial.right_border || pos.second < partial.top_border || pos.second >= partial.bottom_border)
        return -1;
    return partial.cost2go[pos.first - partial.left_border][pos.second - partial.top_border];
}

Encoder::Encoder(const InputParameters &cfg) : cfg(cfg)
{
    for (int i = -cfg.cost2go_value_limit; i <= cfg.cost2go_value_limit; ++i)
        coord_range.push_back(i);
    coord_range.push_back(-cfg.cost2go_value_limit * 4);
    coord_range.push_back(-cfg.cost2go_value_limit * 2);
    coord_range.push_back(cfg.cost2go_value_limit * 2);

    actions_range = {'n', 'w', 'u', 'd', 'l', 'r'};
    for (int i = 0; i < 16; ++i)
    {
        std::stringstream ss;
        ss << std::bitset<4>(i);
        next_action_range.push_back(ss.str());
    }

    int idx = 0;
    for (auto &token : coord_range)
        int_vocab[token] = idx++;
    for (auto &token : actions_range)
        str_vocab[std::string(1, token)] = idx++;
    for (auto &token : next_action_range)
        str_vocab[token] = idx++;
    str_vocab["!"] = idx;

    for (auto &[token, idx] : int_vocab)
        inverse_int_vocab[idx] = token;
    for (auto &[token, idx] : str_vocab)
        inverse_str_vocab[idx] = token;
}

std::vector<int> Encoder::encode(const std::vector<AgentsInfo> &agents, const std::vector<std::vector<int>> &cost2go)
{
    std::vector<int> agents_indices;
    for (const auto &agent : agents)
    {
        std::vector<int> coord_indices = {
            int_vocab.at(agent.relative_pos.first),
            int_vocab.at(agent.relative_pos.second),
            int_vocab.at(std::clamp(agent.relative_goal.first, -cfg.cost2go_value_limit, cfg.cost2go_value_limit)),
            int_vocab.at(std::clamp(agent.relative_goal.second, -cfg.cost2go_value_limit, cfg.cost2go_value_limit))};

        std::vector<int> actions_indices;
        for (const auto &action : agent.previous_actions)
        {
            actions_indices.push_back(str_vocab.at(action));
        }
        std::vector<int> next_action_indices = {str_vocab.at(agent.next_action)};

        agents_indices.insert(agents_indices.end(), coord_indices.begin(), coord_indices.end());
        agents_indices.insert(agents_indices.end(), actions_indices.begin(), actions_indices.end());
        agents_indices.insert(agents_indices.end(), next_action_indices.begin(), next_action_indices.end());
    }

    if (agents.size() < cfg.num_agents)
        agents_indices.insert(agents_indices.end(), (cfg.num_agents - agents.size()) * (5 + cfg.num_previous_actions), str_vocab["!"]);

    std::vector<int> cost2go_indices;
    for (const auto &row : cost2go)
        for (int value : row)
            cost2go_indices.push_back(int_vocab.at(value));

    std::vector<int> result;
    result.insert(result.end(), cost2go_indices.begin(), cost2go_indices.end());
    result.insert(result.end(), agents_indices.begin(), agents_indices.end());
    while (result.size() < 256)
        result.push_back(str_vocab["!"]);
    return result;
}

void ObservationGenerator::create_agents(const std::vector<std::pair<int, int>> &positions, const std::vector<std::pair<int, int>> &goals)
{
    agents.clear();
    int total_agents = positions.size();
    agents.resize(total_agents);
    cost2go_obs_buffer.resize(total_agents, std::vector<std::vector<int>>(2 * cfg.obs_radius + 1, std::vector<int>(2 * cfg.obs_radius + 1)));
    cost2go_partials.resize(total_agents);
#pragma omp parallel for
    for (int i = 0; i < total_agents; i++)
    {
        agents[i].pos = positions[i];
        agents[i].goal = goals[i];
        for (int j = 0; j < cfg.num_previous_actions; ++j)
        {
            agents[i].action_history.push_back("n");
        }
        compute_cost2go_partial(i);
        update_next_action(i);
    }
}

void ObservationGenerator::update_next_action(int agent_idx)
{
    std::string next_action;
    auto &agent = agents[agent_idx];
    std::vector<std::pair<int, int>> moves = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    int current_cost = get_distance(agent_idx, agent.pos);

    for (const auto &move : moves)
    {
        std::pair<int, int> new_pos = {agent.pos.first + move.first, agent.pos.second + move.second};
        int neighbor_cost = get_distance(agent_idx, new_pos);

        if (neighbor_cost >= 0 && current_cost > neighbor_cost)
            next_action += "1";
        else
            next_action += "0";
    }
    agent.next_action = next_action;
}

void ObservationGenerator::update_agents(const std::vector<std::pair<int, int>> &positions, const std::vector<std::pair<int, int>> &goals, const std::vector<int> &actions)
{
    for (const auto &agent : agents)
        agents_locations[agent.pos.first][agent.pos.second] = -1; // first clear old locations for ALL agents
    std::vector<size_t> need_to_update;
    for (size_t i = 0; i < agents.size(); i++)
    {
        auto &agent = agents[i];
        agents_locations[positions[i].first][positions[i].second] = i;
        agent.pos = positions[i];
        switch (actions[i])
        {
        case 0:
            agent.action_history.push_back("w");
            break;
        case 1:
            agent.action_history.push_back("u");
            break;
        case 2:
            agent.action_history.push_back("d");
            break;
        case 3:
            agent.action_history.push_back("l");
            break;
        case 4:
            agent.action_history.push_back("r");
            break;
        default:
            agent.action_history.push_back("n");
            break;
        }
        agent.action_history.pop_front();
        if (agent.goal != goals[i])
        {
            agent.goal = goals[i];
            need_to_update.push_back(i);
        }
        else
        {
            const auto &partial = cost2go_partials[i];
            if (agent.pos.first - cfg.obs_radius < partial.left_border ||
                agent.pos.first + cfg.obs_radius > partial.right_border ||
                agent.pos.second - cfg.obs_radius < partial.top_border ||
                agent.pos.second + cfg.obs_radius > partial.bottom_border)
                need_to_update.push_back(i);
        }
    }

#pragma omp parallel for
    for (size_t i = 0; i < need_to_update.size(); i++)
        compute_cost2go_partial(need_to_update[i]);
    for (size_t i = 0; i < agents.size(); i++)
        update_next_action(i);
}

std::vector<AgentsInfo> ObservationGenerator::get_agents_info(int agent_idx)
{
    std::vector<AgentsInfo> agents_info;
    std::vector<int> considered_agents;
    const auto &cur_agent = agents[agent_idx];
    for (int i = -cfg.agents_radius; i <= cfg.agents_radius; i++)
        for (int j = -cfg.agents_radius; j <= cfg.agents_radius; j++)
            if (agents_locations[cur_agent.pos.first + i][cur_agent.pos.second + j] >= 0)
                considered_agents.push_back(agents_locations[cur_agent.pos.first + i][cur_agent.pos.second + j]);
    std::vector<int> distances(considered_agents.size(), -1);
    for (size_t i = 0; i < considered_agents.size(); i++)
        distances[i] = std::abs(agents[considered_agents[i]].pos.first - cur_agent.pos.first) +
                       std::abs(agents[considered_agents[i]].pos.second - cur_agent.pos.second);
    std::vector<std::pair<int, int>> distance_agent_pairs;
    for (size_t i = 0; i < considered_agents.size(); i++)
    {
        distance_agent_pairs.push_back({distances[i], considered_agents[i]});
    }
    std::sort(distance_agent_pairs.begin(), distance_agent_pairs.end());
    for (int i = 0; i < std::min(int(distance_agent_pairs.size()), cfg.num_agents); i++)
    {
        const auto &agent = agents[distance_agent_pairs[i].second];
        agents_info.push_back(AgentsInfo(std::make_pair(agent.pos.first - cur_agent.pos.first, agent.pos.second - cur_agent.pos.second),
                                         std::make_pair(agent.goal.first - cur_agent.pos.first, agent.goal.second - cur_agent.pos.second),
                                         agent.action_history, agent.next_action));
    }
    return agents_info;
}

std::vector<std::vector<int>> ObservationGenerator::generate_observations()
{

    std::vector<std::vector<int>> observations(agents.size());
#pragma omp parallel for
    for (size_t i = 0; i < agents.size(); i++)
    {
        generate_cost2go_obs(i, false, cost2go_obs_buffer[i]);
        std::vector<AgentsInfo> agents_info = get_agents_info(i);
        observations[i] = encoder.encode(agents_info, cost2go_obs_buffer[i]);
    }
    return observations;
}

int main()
{
    std::vector<std::vector<int>> grid = std::vector<std::vector<int>>(256, std::vector<int>(256, 0));
    ObservationGenerator obs_gen(grid, InputParameters());
    obs_gen.create_agents(std::vector<std::pair<int, int>>{{120, 120}}, std::vector<std::pair<int, int>>{{20, 200}});
    obs_gen.update_agents(std::vector<std::pair<int, int>>{{120, 120}}, std::vector<std::pair<int, int>>{{20, 200}}, std::vector<int>{0});
    auto obs = obs_gen.generate_observations();
    for (const auto &obs_row : obs)
    {
        for (const auto &cell : obs_row)
            std::cout << cell << " ";
        std::cout << std::endl;
    }
    return 0;
}

#ifdef PYBIND11_MODULE
namespace py = pybind11;
PYBIND11_MODULE(observation_generator, m)
{
    py::class_<InputParameters>(m, "InputParameters")
        .def(py::init<int, int, int, int, int, int, int, bool>())
        .def_readwrite("cost2go_value_limit", &InputParameters::cost2go_value_limit)
        .def_readwrite("num_agents", &InputParameters::num_agents)
        .def_readwrite("num_previous_actions", &InputParameters::num_previous_actions)
        .def_readwrite("agents_radius", &InputParameters::agents_radius)
        .def_readwrite("context_size", &InputParameters::context_size)
        .def_readwrite("obs_radius", &InputParameters::obs_radius);
    py::class_<ObservationGenerator>(m, "ObservationGenerator")
        .def(py::init<const std::vector<std::vector<int>> &, const InputParameters &>())
        .def("create_agents", &ObservationGenerator::create_agents)
        .def("update_agents", &ObservationGenerator::update_agents)
        .def("generate_observations", &ObservationGenerator::generate_observations);
}
<%
import platform
import os

# Cross-platform OpenMP configuration
if platform.system() == 'Darwin':  # macOS
    if os.path.exists('/opt/homebrew/opt/libomp'):  # Apple Silicon (M1/M2)
        cfg['extra_compile_args'] = ['-std=c++17', '-Xpreprocessor', '-fopenmp', '-m64', '-I/opt/homebrew/opt/libomp/include']
        cfg['extra_link_args'] = ['-L/opt/homebrew/opt/libomp/lib', '-lomp', '-m64']
    elif os.path.exists('/usr/local/opt/libomp'):  # Intel Mac
        cfg['extra_compile_args'] = ['-std=c++17', '-Xpreprocessor', '-fopenmp', '-m64', '-I/usr/local/opt/libomp/include']
        cfg['extra_link_args'] = ['-L/usr/local/opt/libomp/lib', '-lomp', '-m64']
    else:
        # Fallback for macOS without homebrew OpenMP
        cfg['extra_compile_args'] = ['-std=c++17', '-m64']
        cfg['extra_link_args'] = ['-m64']
else:  # Linux and other Unix systems
    cfg['extra_compile_args'] = ['-std=c++17', '-fopenmp', '-m64']
    cfg['extra_link_args'] = ['-fopenmp', '-m64']

setup_pybind11(cfg)
%>
#endif