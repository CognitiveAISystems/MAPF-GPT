#include "../include/dist_table.hpp"

DistTable::DistTable(const Instance &ins)
    : K(ins.G->V.size()), table(ins.N, std::vector<int>(K, K))
{
  setup(&ins);
}

DistTable::DistTable(const Instance *ins)
    : K(ins->G->V.size()), table(ins->N, std::vector<int>(K, K))
{
  setup(ins);
}

void DistTable::setup(const Instance *ins)
{
  auto bfs = [&](const int i) {
    auto g_i = ins->goals[i];
    auto Q = std::queue<Vertex *>({g_i});
    table[i][g_i->id] = 0;
    while (!Q.empty()) {
      auto n = Q.front();
      Q.pop();
      const int d_n = table[i][n->id];
      for (auto &m : n->neighbor) {
        const int d_m = table[i][m->id];
        if (d_n + 1 >= d_m) continue;
        table[i][m->id] = d_n + 1;
        Q.push(m);
      }
    }
  };

  auto pool = std::vector<std::future<void>>();
  for (size_t i = 0; i < ins->N; ++i) {
    pool.emplace_back(std::async(std::launch::async, bfs, i));
  }
}

int DistTable::get(const int i, const int v_id) { return table[i][v_id]; }

int DistTable::get(const int i, const Vertex *v) { return get(i, v->id); }
