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
#define PYBIND11_MODULE
#ifdef PYBIND11_MODULE
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#endif
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
    std::vector<Agent> agents;
    InputParameters cfg;
    Encoder encoder;
    std::vector<std::vector<int>> agents_locations;
    std::vector<std::vector<std::vector<int>>> cost2go_obs_buffer; // Buffer for each agent
    std::vector<std::vector<int>> grid;
    std::vector<std::vector<int>> components;
    std::vector<Cost2GoPartial> cost2go_partials;
    ObservationGenerator(const std::vector<std::vector<int>> &grid, const InputParameters &cfg)
        : grid(grid), cfg(cfg), encoder(cfg)
    {
        agents_locations = std::vector<std::vector<int>>(grid.size(), std::vector<int>(grid[0].size(), -1));
        mark_components();
    }
    ~ObservationGenerator() {}
    void mark_components();
    void compute_cost2go_partial(int agent_idx);
    void generate_cost2go_obs(int agent_idx, bool only_obstacles, std::vector<std::vector<int>> &buffer);
    int get_distance(int agent_idx, const std::pair<int, int> &pos);

    void create_agents(const std::vector<std::pair<int, int>> &positions, const std::vector<std::pair<int, int>> &goals);
    void update_next_action(int agent_idx);
    void update_agents(const std::vector<std::pair<int, int>> &positions, const std::vector<std::pair<int, int>> &goals, const std::vector<int> &actions);
    std::vector<AgentsInfo> get_agents_info(int agent_idx);
    std::vector<std::vector<int>> generate_observations();
};