/*
 * Implementation of SUO
 *
 * references:
 * Optimizingspaceutilizationformoreeffective multi-robot path planning.
 * Shuai D Han and Jingjin Yu.
 * In Proceedings of IEEE International Conference on Robotics and Automation
 * (ICRA). 2022.
 */
#pragma once

#include "collision_table.hpp"
#include "dist_table.hpp"
#include "graph.hpp"
#include "utils.hpp"

struct Scatter {
  const Instance *ins;
  const Deadline *deadline;
  std::mt19937 MT;
  const int verbose;
  const int N;
  const int V_size;
  const int T;  // makespan lower bound
  DistTable *D;
  const int cost_margin;
  int sum_of_path_length;

  // outcome
  std::vector<Path> paths;
  // agent, vertex-id, next vertex
  std::vector<std::unordered_map<int, Vertex *>> scatter_data;

  // collision data
  CollisionTable CT;

  void construct();

  Scatter(const Instance *_ins, DistTable *_D, const Deadline *_deadline,
          const int seed = 0, int _verbose = 0, int _cost_margin = 2);
};
