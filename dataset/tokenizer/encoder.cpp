// cppimport
#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <bitset>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

struct InputParameters {
    InputParameters(int cvl, int na, int npa, int cs):
    cost2go_value_limit(cvl), num_agents(na), num_previous_actions(npa), context_size(cs) {}
    int cost2go_value_limit;
    int num_agents;
    int num_previous_actions;
    int context_size = 256;
};

struct AgentsInfo
{
    AgentsInfo(std::pair<int, int> rp, std::pair<int, int> rg, std::vector<std::string> pa, std::string na):
    relative_pos(rp), relative_goal(rg), previous_actions(pa), next_action(na) {}
    std::pair<int, int> relative_pos;
    std::pair<int, int> relative_goal;
    std::vector<std::string> previous_actions;
    std::string next_action;
};

std::string to_repr(const AgentsInfo& self) {
    std::ostringstream oss;
    oss << "(relative_pos=(" << self.relative_pos.first << ", " << self.relative_pos.second
        << "), relative_goal=(" << self.relative_goal.first << ", " << self.relative_goal.second
        << "), previous_actions=[";

    for (size_t i = 0; i < self.previous_actions.size(); ++i) {
        oss << self.previous_actions[i];
        if (i < self.previous_actions.size() - 1) {
            oss << ", ";
        }
    }
    oss << "], next_action=" << self.next_action << ")";

    return oss.str();
}

class Encoder {
public:
    Encoder(const InputParameters& cfg)
        : cfg(cfg) {
        for (int i = -cfg.cost2go_value_limit; i <= cfg.cost2go_value_limit; ++i) {
            coord_range.push_back(i);
        }
        coord_range.push_back(-cfg.cost2go_value_limit * 4);
        coord_range.push_back(-cfg.cost2go_value_limit * 2);
        coord_range.push_back(cfg.cost2go_value_limit * 2);

        actions_range = {'n', 'w', 'u', 'd', 'l', 'r'};
        for (int i = 0; i < 16; ++i) {
            std::stringstream ss;
            ss << std::bitset<4>(i);
            next_action_range.push_back(ss.str());
        }

        int idx = 0;
        for (auto& token : coord_range) {
            int_vocab[token] = idx++;
        }
        for (auto& token : actions_range) {
            str_vocab[std::string(1, token)] = idx++;
        }
        for (auto& token : next_action_range) {
            str_vocab[token] = idx++;
        }
        str_vocab["!"] = idx;

        for (auto& [token, idx] : int_vocab) {
            inverse_int_vocab[idx] = token;
        }
        for (auto& [token, idx] : str_vocab) {
            inverse_str_vocab[idx] = token;
        }
    }

    std::vector<int> encode(const std::vector<AgentsInfo>& agents, const std::vector<std::vector<int>> &cost2go) {
        std::vector<int> agents_indices;
        for (const auto& agent : agents) {
            std::vector<int> coord_indices = {
                int_vocab.at(agent.relative_pos.first),
                int_vocab.at(agent.relative_pos.second),
                int_vocab.at(agent.relative_goal.first),
                int_vocab.at(agent.relative_goal.second)
            };

            std::vector<int> actions_indices;
            for (const auto& action : agent.previous_actions) {
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
        for (const auto& row : cost2go)
            for (int value : row)
                cost2go_indices.push_back(int_vocab.at(value));

        std::vector<int> result;
        result.insert(result.end(), cost2go_indices.begin(), cost2go_indices.end());
        result.insert(result.end(), agents_indices.begin(), agents_indices.end());
        while(result.size() < 256)
            result.push_back(str_vocab["!"]);
        return result;
    }

private:
    InputParameters cfg;
    std::vector<int> coord_range;
    std::vector<char> actions_range;
    std::vector<std::string> next_action_range;
    std::unordered_map<std::string, int> str_vocab;
    std::unordered_map<int, int> int_vocab;
    std::unordered_map<int, int> inverse_int_vocab;
    std::unordered_map<int, std::string> inverse_str_vocab;

    std::string join(const std::vector<std::string>& vec, const std::string& delim) {
        std::ostringstream res;
        copy(vec.begin(), vec.end(), std::ostream_iterator<std::string>(res, delim.c_str()));
        return res.str().substr(0, res.str().length() - delim.length());
    }
};

PYBIND11_MODULE(encoder, m) {
    py::class_<InputParameters>(m, "InputParameters")
        .def(py::init<int, int, int, int>())
        .def_readwrite("cost2go_value_limit", &InputParameters::cost2go_value_limit)
        .def_readwrite("num_agents", &InputParameters::num_agents)
        .def_readwrite("num_previous_actions", &InputParameters::num_previous_actions)
        ;

    py::class_<AgentsInfo>(m, "AgentsInfo")
        .def(py::init<std::pair<int, int>, std::pair<int, int>, std::vector<std::string>, std::string>())
        .def_readwrite("relative_pos", &AgentsInfo::relative_pos)
        .def_readwrite("relative_goal", &AgentsInfo::relative_goal)
        .def_readwrite("previous_actions", &AgentsInfo::previous_actions)
        .def_readwrite("next_action", &AgentsInfo::next_action)
        .def("__repr__", &to_repr);
        ;

    py::class_<Encoder>(m, "Encoder")
        .def(py::init<const InputParameters&>())
        .def("encode", &Encoder::encode)
        ;
}

<%
cfg['extra_compile_args'] = ['-std=c++17']
setup_pybind11(cfg)
%>
