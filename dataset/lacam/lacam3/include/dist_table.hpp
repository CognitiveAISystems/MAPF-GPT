/*
 * distance table with lazy evaluation, using BFS
 */
#pragma once

#include "graph.hpp"
#include "instance.hpp"
#include "utils.hpp"

struct DistTable {
  const int K;  // number of vertices
  std::vector<std::vector<int>>
      table;  // distance table, index: agent-id & vertex-id
  std::vector<std::queue<Vertex *>> OPEN;  // search queue

  int get(const int i, const int v_id);   // agent, vertex-id
  int get(const int i, const Vertex *v);  // agent, vertex

  DistTable(const Instance &ins);
  DistTable(const Instance *ins);

  void setup(const Instance *ins);  // initialization
};
