#include "../include/collision_table.hpp"

CollisionTable::CollisionTable(const Instance *ins)
    : body(ins->G->size()),
      body_last(ins->G->size()),
      collision_cnt(0),
      N(ins->N)
{
}

CollisionTable::~CollisionTable() {}

int CollisionTable::getCollisionCost(const Vertex *v_from, const Vertex *v_to,
                                     const int t_from)
{
  const int t_to = t_from + 1;
  auto collision = 0;
  // vertex collision
  if (t_to < body[v_to->id].size()) {
    collision += body[v_to->id][t_to].size();
  }
  // edge collision
  if (t_to < body[v_from->id].size() && t_from < body[v_to->id].size()) {
    for (auto j : body[v_from->id][t_to]) {
      for (auto k : body[v_to->id][t_from]) {
        if (j == k) ++collision;
      }
    }
  }
  // goal collision
  for (auto last_timestep : body_last[v_to->id]) {
    if (t_to > last_timestep) ++collision;
  }
  return collision;
}

void CollisionTable::enrollPath(const int i, Path &path)
{
  if (path.empty()) return;
  const auto T_i = path.size() - 1;
  for (auto t = 0; t <= T_i; ++t) {
    auto v = path[t];

    // update collision count
    if (t > 0) collision_cnt += getCollisionCost(path[t - 1], path[t], t - 1);

    // register
    while (body[v->id].size() <= t) body[v->id].emplace_back();
    body[v->id][t].push_back(i);
  }

  // goal
  body_last[path.back()->id].push_back(T_i);
  auto &&entry = body[path.back()->id];
  for (auto t = T_i + 1; t < entry.size(); ++t) {
    collision_cnt += entry[t].size();
  }
}

void CollisionTable::clearPath(const int i, Path &path)
{
  if (path.empty()) return;
  const auto T_i = (int)path.size() - 1;
  for (auto t = 0; t <= T_i; ++t) {
    auto v = path[t];
    auto &&entry = body[v->id][t];

    // remove entry
    for (auto itr = entry.begin(); itr != entry.end();) {
      if (*itr == i) {
        entry.erase(itr);
        break;
      } else {
        ++itr;
      }
    }

    // update collision count
    if (t > 0) collision_cnt -= getCollisionCost(path[t - 1], path[t], t - 1);
  }

  // goal
  auto &&entry_body_last = body_last[path.back()->id];
  for (auto itr = entry_body_last.begin(); itr != entry_body_last.end();) {
    if (*itr == T_i) {
      entry_body_last.erase(itr);
      break;
    } else {
      ++itr;
    }
  }
  auto &&entry_body = body[path.back()->id];
  for (auto t = T_i + 1; t < entry_body.size(); ++t) {
    collision_cnt -= entry_body[t].size();
  }
}
