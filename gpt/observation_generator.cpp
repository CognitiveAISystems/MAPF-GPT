// cppimport
#include "observation_generator.h"

void Cost2GoPartialManager::mark_components()
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

void Cost2GoPartialManager::compute_cost2go_partial(const std::pair<int, int> &goal, const std::pair<int, int> &center)
{
    Cost2GoPartial partial(goal, center);
    std::vector<std::vector<int>> cost2go(2 * cost2go_radius + 1, std::vector<int>(2 * cost2go_radius + 1, -1));
    std::vector<std::vector<int>> cost_matrix(grid.size(), std::vector<int>(grid[0].size(), -1));
    std::set<std::pair<int, int>> cells_in_component;
    for (int i = -cost2go_radius; i <= cost2go_radius; i++)
        for (int j = -cost2go_radius; j <= cost2go_radius; j++)
        {
            int ci = center.first + i;
            int cj = center.second + j;
            if (ci >= 0 && ci < grid.size() && cj >= 0 && cj < grid[0].size() && components[ci][cj] == components[goal.first][goal.second])
                cells_in_component.insert({ci, cj});
        }
    std::vector<std::pair<int, int>> moves = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    std::queue<std::pair<int, int>> fringe;
    fringe.push(goal);
    cost_matrix[goal.first][goal.second] = 0;
    while (!fringe.empty())
    {
        auto pos = fringe.front();
        fringe.pop();
        if (pos.first < center.first + cost2go_radius &&
            pos.first >= center.first - cost2go_radius &&
            pos.second < center.second + cost2go_radius &&
            pos.second >= center.second - cost2go_radius)
        {
            cells_in_component.erase(pos);
            if (cells_in_component.empty())
                break;
        }
        for (const auto &move : moves)
        {
            int new_i(pos.first + move.first), new_j(pos.second + move.second);
            if (new_i >= 0 && new_j >= 0 && new_i < grid.size() && new_j < grid.front().size())
            {
                if (grid[new_i][new_j] == 0 && cost_matrix[new_i][new_j] < 0)
                {
                    cost_matrix[new_i][new_j] = cost_matrix[pos.first][pos.second] + 1;
                    fringe.push(std::make_pair(new_i, new_j));
                }
            }
        }
    }
    for (int i = -cost2go_radius; i <= cost2go_radius; i++)
        for (int j = -cost2go_radius; j <= cost2go_radius; j++)
        {
            int ci = center.first + i;
            int cj = center.second + j;
            if (ci >= 0 && ci < grid.size() && cj >= 0 && cj < grid[0].size())
                cost2go[i + cost2go_radius][j + cost2go_radius] = cost_matrix[ci][cj];
        }
    partial.cost2go = cost2go;
    cost2go_partials[goal] = partial;
}

void Cost2GoPartialManager::generate_cost2go_obs(const std::pair<int, int> &goal, const std::pair<int, int> &pos, bool only_obstacles, std::vector<std::vector<int>> &buffer)
{
    const auto &partial = cost2go_partials[goal];
    int x = pos.first + cost2go_radius - partial.center.first - obs_radius;
    int y = pos.second + cost2go_radius - partial.center.second - obs_radius;
    if (only_obstacles)
        for (int i = 0; i <= obs_radius * 2; i++)
            for (int j = 0; j <= obs_radius * 2; j++)
                buffer[i][j] = bool(partial.cost2go[x + i][y + j] < 0);
    int middle_value = partial.cost2go[x + obs_radius][y + obs_radius];
    for (int i = 0; i <= obs_radius * 2; i++)
        for (int j = 0; j <= obs_radius * 2; j++)
        {
            int value = partial.cost2go[x + i][y + j];
            if (value >= 0)
            {
                value -= middle_value;
                buffer[i][j] = value > limit ? limit * 2 : value < -limit ? -limit * 2
                                                                          : value;
            }
            else
                buffer[i][j] = -limit * 4;
        }
}

int Cost2GoPartialManager::get_distance(const std::pair<int, int> &goal, const std::pair<int, int> &pos)
{
    if (abs(cost2go_partials[goal].center.first - pos.first) > cost2go_radius || abs(cost2go_partials[goal].center.second - pos.second) > cost2go_radius)
        return -1;
    return cost2go_partials[goal].cost2go[pos.first - cost2go_partials[goal].center.first + cost2go_radius][pos.second - cost2go_partials[goal].center.second + cost2go_radius];
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
    int total_agents = positions.size();
    agents.resize(total_agents);
    cost2go_obs_buffer.resize(total_agents, std::vector<std::vector<int>>(2 * cfg.obs_radius + 1, std::vector<int>(2 * cfg.obs_radius + 1)));

    for (int i = 0; i < total_agents; i++)
    {
        agents[i].pos = positions[i];
        agents[i].goal = goals[i];
        for (int j = 0; j < cfg.num_previous_actions; ++j)
        {
            agents[i].action_history.push_back("n");
        }
        manager.compute_cost2go_partial(goals[i], positions[i]);
        agents[i].next_action = get_next_action(agents[i]);
    }
}

std::string ObservationGenerator::get_next_action(const Agent &agent)
{
    std::string next_action;
    std::vector<std::pair<int, int>> moves = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    int current_cost = manager.get_distance(agent.goal, agent.pos);

    for (const auto &move : moves)
    {
        std::pair<int, int> new_pos = {agent.pos.first + move.first, agent.pos.second + move.second};
        int neighbor_cost = manager.get_distance(agent.goal, new_pos);

        if (neighbor_cost >= 0 && current_cost > neighbor_cost)
            next_action += "1";
        else
            next_action += "0";
    }

    return next_action;
}

void ObservationGenerator::update_agents(const std::vector<std::pair<int, int>> &positions, const std::vector<std::pair<int, int>> &goals, const std::vector<int> &actions)
{
    for (const auto &agent : agents)
        agents_locations[agent.pos.first][agent.pos.second] = -1; // first clear old locations for ALL agents
    for (int i = 0; i < agents.size(); i++)
    {
        agents_locations[positions[i].first][positions[i].second] = i;
        agents[i].pos = positions[i];
        switch (actions[i])
        {
        case 0:
            agents[i].action_history.push_back("w");
            break;
        case 1:
            agents[i].action_history.push_back("u");
            break;
        case 2:
            agents[i].action_history.push_back("d");
            break;
        case 3:
            agents[i].action_history.push_back("l");
            break;
        case 4:
            agents[i].action_history.push_back("r");
            break;
        default:
            agents[i].action_history.push_back("n");
            break;
        }
        agents[i].action_history.pop_front();
        if (agents[i].goal != goals[i])
        {
            manager.cost2go_partials.erase(agents[i].goal);
            agents[i].goal = goals[i];
        }
        agents[i].next_action = get_next_action(agents[i]);
    }
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
    for (int i = 0; i < considered_agents.size(); i++)
        distances[i] = std::abs(agents[considered_agents[i]].pos.first - cur_agent.pos.first) +
                       std::abs(agents[considered_agents[i]].pos.second - cur_agent.pos.second);
    std::vector<std::pair<int, int>> distance_agent_pairs;
    for (int i = 0; i < considered_agents.size(); i++)
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
    for (const auto &agent : agents)
    {
        if (manager.cost2go_partials.find(agent.goal) == manager.cost2go_partials.end())
        {
            manager.compute_cost2go_partial(agent.goal, agent.pos);
        }
        const auto &partial = manager.cost2go_partials[agent.goal];
        if (abs(agent.pos.first - partial.center.first) > manager.cost2go_radius - cfg.obs_radius ||
            abs(agent.pos.second - partial.center.second) > manager.cost2go_radius - cfg.obs_radius)
        {
            manager.compute_cost2go_partial(agent.goal, agent.pos);
        }
    }

    std::vector<std::vector<int>> observations(agents.size());
    for (int i = 0; i < agents.size(); i++)
    {
        manager.generate_cost2go_obs(agents[i].goal, agents[i].pos, false, cost2go_obs_buffer[i]);
        std::vector<AgentsInfo> agents_info = get_agents_info(i);
        observations[i] = encoder.encode(agents_info, cost2go_obs_buffer[i]);
    }
    return observations;
}

/*
int main()
{
    std::vector<std::vector<int>> grid = std::vector<std::vector<int>>(50, std::vector<int>(50, 0));
    ObservationGenerator obs_gen(grid, InputParameters());
    obs_gen.create_agents(std::vector<std::pair<int, int>>{{23, 23}}, std::vector<std::pair<int, int>>{{20, 20}});
    obs_gen.update_agents(std::vector<std::pair<int, int>>{{23, 23}}, std::vector<std::pair<int, int>>{{20, 20}}, std::vector<int>{0});
    auto obs = obs_gen.generate_observations();
    for(const auto &obs_row : obs)
    {
        for(const auto &cell : obs_row)
            std::cout << cell << " ";
        std::cout << std::endl;
    }
    //std::cout << manager.get_distance(std::make_pair(23, 23), std::make_pair(20, 20)) << std::endl;
    return 0;
}
*/

namespace py = pybind11;

PYBIND11_MODULE(observation_generator, m)
{
    py::class_<InputParameters>(m, "InputParameters")
        .def(py::init<int, int, int, int, int, int, int>())
        .def_readwrite("cost2go_value_limit", &InputParameters::cost2go_value_limit)
        .def_readwrite("num_agents", &InputParameters::num_agents)
        .def_readwrite("num_previous_actions", &InputParameters::num_previous_actions)
        .def_readwrite("agents_radius", &InputParameters::agents_radius)
        .def_readwrite("cost2go_radius", &InputParameters::cost2go_radius)
        .def_readwrite("context_size", &InputParameters::context_size)
        .def_readwrite("obs_radius", &InputParameters::obs_radius);
    py::class_<ObservationGenerator>(m, "ObservationGenerator")
        .def(py::init<const std::vector<std::vector<int>> &, const InputParameters &>())
        .def("create_agents", &ObservationGenerator::create_agents)
        .def("update_agents", &ObservationGenerator::update_agents)
        .def("generate_observations", &ObservationGenerator::generate_observations);
}
/*
<%
cfg['extra_compile_args'] = ['-std=c++17']
setup_pybind11(cfg)
%>
*/