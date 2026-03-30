#include "../include/scatter.hpp"

#include "../include/metrics.hpp"

Scatter::Scatter(const Instance *_ins, DistTable *_D, const Deadline *_deadline,
                 const int seed, int _verbose, int _cost_margin)
    : ins(_ins),
      deadline(_deadline),
      MT(std::mt19937(seed)),
      verbose(_verbose),
      N(ins->N),
      V_size(ins->G->size()),
      T(get_makespan_lower_bound(*ins, *_D) + _cost_margin),
      D(_D),
      cost_margin(_cost_margin),
      sum_of_path_length(0),
      paths(N),
      scatter_data(N),
      CT(ins)
{
}

void Scatter::construct()
{
  info(0, verbose, deadline, "scatter", "\tinvoked");

  // define path finding utilities
  // vertex, cost-to-come, cost-to-go, collision, parent
  using ScatterNode = std::tuple<Vertex *, int, int, int, Vertex *>;
  auto cmp = [&](ScatterNode &a, ScatterNode &b) {
    // collision
    if (std::get<3>(a) != std::get<3>(b))
      return std::get<3>(a) > std::get<3>(b);
    auto f_a = std::get<1>(a) + std::get<2>(a);
    auto f_b = std::get<1>(b) + std::get<2>(b);
    if (f_a != f_b) return f_a > f_b;
    return std::get<0>(a)->id < std::get<0>(b)->id;
  };
  auto CLOSED = std::vector<Vertex *>(V_size, nullptr);  // parent

  // metrics
  auto collision_cnt_last = 0;
  auto paths_prev = std::vector<Path>();

  // main loop
  auto loop = 0;
  while (loop < 2 || CT.collision_cnt < collision_cnt_last) {
    ++loop;
    collision_cnt_last = CT.collision_cnt;

    // randomize planning order
    auto order = std::vector<int>(N, 0);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), MT);

    // single-agent path finding for agent-i
    for (int _i = 0; _i < N; ++_i) {
      if (is_expired(deadline)) break;

      const auto i = order[_i];
      const auto cost_ub = D->get(i, ins->starts[i]) + cost_margin;

      if (!paths[i].empty()) sum_of_path_length -= (paths[i].size() - 1);

      // clear cache
      CT.clearPath(i, paths[i]);

      // setup A*
      auto OPEN = std::priority_queue<ScatterNode, std::vector<ScatterNode>,
                                      decltype(cmp)>(cmp);
      // used with CLOSED, vertex-id list
      const auto s_i = ins->starts[i];
      OPEN.push(std::make_tuple(s_i, 0, D->get(i, s_i), 0, nullptr));
      auto USED = std::vector<int>();

      // A*
      auto solved = false;
      while (!OPEN.empty() && !is_expired(deadline)) {
        // pop
        auto node = OPEN.top();
        OPEN.pop();

        // check CLOSED list
        const auto v = std::get<0>(node);
        const auto g_v = std::get<1>(node);  // cost-to-come
        const auto c_v = std::get<3>(node);  // collision
        if (CLOSED[v->id] != nullptr) continue;
        CLOSED[v->id] = std::get<4>(node);  // parent
        USED.push_back(v->id);

        // check goal condition
        if (v == ins->goals[i]) {
          solved = true;
          break;
        }

        // expand
        for (auto u : v->neighbor) {
          auto d_u = D->get(i, u);
          if (u != s_i && CLOSED[u->id] == nullptr &&
              d_u + g_v + 1 <= cost_ub) {
            // insert new node
            OPEN.push(std::make_tuple(u, g_v + 1, d_u,
                                      CT.getCollisionCost(v, u, g_v) + c_v, v));
          }
        }
      }

      // backtrack
      if (solved) {
        paths[i].clear();
        auto v = ins->goals[i];
        while (v != nullptr) {
          paths[i].push_back(v);
          v = CLOSED[v->id];
        }
        std::reverse(paths[i].begin(), paths[i].end());
      }

      // register to CT & update collision count
      CT.enrollPath(i, paths[i]);
      sum_of_path_length += paths[i].size() - 1;

      // memory management
      for (auto k : USED) CLOSED[k] = nullptr;
    }

    paths_prev = paths;
    info(1, verbose, deadline, "scatter", "\titer:", loop,
         "\tcollision_cnt:", CT.collision_cnt);

    if (CT.collision_cnt == 0) break;
    if (is_expired(deadline)) break;
  }

  paths = paths_prev;

  // set scatter data
  for (auto i = 0; i < N; ++i) {
    if (paths[i].empty()) continue;
    for (auto t = 0; t < paths[i].size() - 1; ++t) {
      scatter_data[i][paths[i][t]->id] = paths[i][t + 1];
    }
  }

  info(0, verbose, deadline, "scatter", "\tcompleted");
}
