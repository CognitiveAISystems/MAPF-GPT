// cppimport
#include <vector>
#include <queue>
#include <map>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

std::vector<std::vector<int>> get_cost_matrix(const std::vector<std::vector<int>> &grid, int si, int sj)
{
    std::vector<std::pair<int, int>> moves = {{0, 0}, {-1, 0}, {1, 0}, {0, -1}, {0, 1}};
    std::queue<std::pair<int, int>> fringe;
    fringe.push({si, sj});
    auto result = std::vector<std::vector<int>>(grid.size(), std::vector<int>(grid[0].size(), -1));
    result[si][sj] = 0;
    while (!fringe.empty())
    {
        auto pos = fringe.front();
        fringe.pop();
        for (const auto &move : moves)
        {
            int new_i(pos.first + move.first), new_j(pos.second + move.second);
            if(new_i >=0 && new_j >= 0 && new_i < grid.size() && new_j < grid.front().size())
                if (grid[new_i][new_j] == 0 && result[new_i][new_j] < 0)
                {
                    result[new_i][new_j] = result[pos.first][pos.second] + 1;
                    fringe.push(std::make_pair(new_i, new_j));
                }
        }
    }
    return result;
}

std::map<std::pair<int, int>, std::vector<std::vector<int>>> precompute_cost2go(const std::vector<std::vector<int>> &grid, int obs_radius)
{
    std::map<std::pair<int, int>, std::vector<std::vector<int>>> cost2go;
    for (size_t i = obs_radius; i < grid.size() - obs_radius; i++)
        for (size_t j = obs_radius; j < grid[0].size() - obs_radius; j++)
            if (grid[i][j] == 0)
                cost2go[std::make_pair(i, j)] = get_cost_matrix(grid, i, j);
    return cost2go;
}

std::vector<std::vector<int>> generate_cost2go_obs(const std::vector<std::vector<int>> &cost2go, const std::pair<int, int> &pos, int offset, int limit, bool only_obstacles)
{
    if (offset == 0)
        return {};
    int x = pos.first - offset;
    int y = pos.second - offset;

    std::vector<std::vector<int>> observation(2 * offset + 1, std::vector<int>(2 * offset + 1));
    if (only_obstacles)
    {
        for (int i = 0; i <= offset * 2; i++)
        for (int j = 0; j <= offset * 2; j++)
        {
            int nx = x + i;
            int ny = y + j;
            observation[i][j] = bool(cost2go[nx][ny] < 0);
        }
        return observation;
    }
    for (int i = 0; i <= offset * 2; i++)
        for (int j = 0; j <= offset * 2; j++)
        {
            int nx = x + i;
            int ny = y + j;
            observation[i][j] = cost2go[nx][ny];
        }

    int middle_value = observation[offset][offset];
    for (int i = 0; i < observation.size(); i++)
        for (int j = 0; j < observation[i].size(); j++)
        {
            if (observation[i][j] >= 0)
            {
                observation[i][j] -= middle_value;
                if (observation[i][j] > limit)
                    observation[i][j] = limit * 2;
                else if (observation[i][j] < -limit)
                    observation[i][j] = -limit * 2;
            }
            else
                observation[i][j] = -limit * 4;
        }

    return observation;
}

namespace py = pybind11;

PYBIND11_MODULE(cost2go, m)
{
    m.def("precompute_cost2go", &precompute_cost2go, "Precompute cost-to-go matrices for the grid",
          py::arg("grid"), py::arg("obs_radius"));
    m.def("get_cost_matrix", &get_cost_matrix, "Compute cost matrix from a starting position",
          py::arg("grid"), py::arg("si"), py::arg("sj"));
    m.def("generate_cost2go_obs", &generate_cost2go_obs, "Generate cost-to-go observations",
          py::arg("cost2go"), py::arg("pos"), py::arg("offset"), py::arg("limit"), py::arg("only_obstacles"));
}

<%
cfg['extra_compile_args'] = ['-std=c++17']
setup_pybind11(cfg)
%>