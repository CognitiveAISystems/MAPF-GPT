// cppimport
#pragma once
#include <vector>
#include <queue>
#include <map>
#include <set>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <bitset>
#include <algorithm>
#include <sstream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

struct InputParameters
{
    InputParameters(int cvl = 20, int na = 13, int npa = 5, int cs = 256, int obsr = 5, int cr = 10, int ar = 5) : cost2go_value_limit(cvl),
                                                                                                                   num_agents(na),
                                                                                                                   num_previous_actions(npa),
                                                                                                                   context_size(cs),
                                                                                                                   obs_radius(obsr),
                                                                                                                   cost2go_radius(cr),
                                                                                                                   agents_radius(ar) {}
    int cost2go_value_limit;
    int num_agents;
    int num_previous_actions;
    int context_size;
    int obs_radius;
    int cost2go_radius;
    int agents_radius;
};

struct AgentsInfo
{
    AgentsInfo(std::pair<int, int> rp, std::pair<int, int> rg, std::deque<std::string> pa, std::string na) : relative_pos(rp), relative_goal(rg), previous_actions(pa), next_action(na) {}
    AgentsInfo() {}
    std::pair<int, int> relative_pos;
    std::pair<int, int> relative_goal;
    std::deque<std::string> previous_actions;
    std::string next_action;
};

struct Agent
{
    std::pair<int, int> pos;
    std::pair<int, int> goal;
    std::deque<std::string> action_history;
    std::string next_action;
};

struct Cost2GoPartial
{
    std::pair<int, int> goal;
    std::pair<int, int> center;
    std::vector<std::vector<int>> cost2go;
    Cost2GoPartial(const std::pair<int, int> &goal = std::make_pair(-1, -1),
                   const std::pair<int, int> &center = std::make_pair(-1, -1)) : goal(goal), center(center)
    {}
};

class Cost2GoPartialManager
{
public:
    std::vector<std::vector<int>> grid;
    std::vector<std::vector<int>> components;
    std::map<std::pair<int, int>, Cost2GoPartial> cost2go_partials;
    int obs_radius;
    int cost2go_radius;
    int limit;
    Cost2GoPartialManager(const std::vector<std::vector<int>> &grid, int obs_radius, int cost2go_radius, int limit) : grid(grid), obs_radius(obs_radius), cost2go_radius(cost2go_radius), limit(limit)
    {
        mark_components();
    }
    void mark_components();
    void compute_cost2go_partial(const std::pair<int, int> &goal, const std::pair<int, int> &center);
    void generate_cost2go_obs(const std::pair<int, int> &goal, const std::pair<int, int> &pos, bool only_obstacles, std::vector<std::vector<int>> &buffer);
    int get_distance(const std::pair<int, int> &goal, const std::pair<int, int> &pos);
};

class Encoder
{
public:
    InputParameters cfg;
    std::vector<int> coord_range;
    std::vector<char> actions_range;
    std::vector<std::string> next_action_range;
    std::unordered_map<std::string, int> str_vocab;
    std::unordered_map<int, int> int_vocab;
    std::unordered_map<int, int> inverse_int_vocab;
    std::unordered_map<int, std::string> inverse_str_vocab;
    Encoder(const InputParameters &cfg);
    std::vector<int> encode(const std::vector<AgentsInfo> &agents, const std::vector<std::vector<int>> &cost2go);
};

class ObservationGenerator
{
public:
    Cost2GoPartialManager manager;
    std::vector<Agent> agents;
    InputParameters cfg;
    Encoder encoder;
    std::vector<std::vector<int>> agents_locations;
    std::vector<std::vector<std::vector<int>>> cost2go_obs_buffer; // Buffer for each agent

    ObservationGenerator(const std::vector<std::vector<int>> &grid, const InputParameters &cfg)
        : manager(grid, cfg.obs_radius, cfg.cost2go_radius, cfg.cost2go_value_limit), cfg(cfg), encoder(cfg)
    {
        agents_locations = std::vector<std::vector<int>>(grid.size(), std::vector<int>(grid[0].size(), -1));
    }

    void create_agents(const std::vector<std::pair<int, int>> &positions, const std::vector<std::pair<int, int>> &goals);
    std::string get_next_action(const Agent &agent);
    void update_agents(const std::vector<std::pair<int, int>> &positions, const std::vector<std::pair<int, int>> &goals, const std::vector<int> &actions);
    std::vector<AgentsInfo> get_agents_info(int agent_idx);
    std::vector<std::vector<int>> generate_observations();
};