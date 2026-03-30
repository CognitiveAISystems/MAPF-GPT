/*
 * implementation of SIPP
 *
 * references:
 * Sipp: Safe interval path planning for dynamic environments.
 * Mike Phillips and Maxim Likhachev.
 * In Proceedings of IEEE International Conference on Robotics and Automation
 * (ICRA). 2011.
 */

#pragma once

#include "collision_table.hpp"
#include "dist_table.hpp"
#include "graph.hpp"
#include "utils.hpp"

// safe interval
using SI = std::pair<int, int>;
using SIs = std::vector<SI>;

struct SITable {
  std::unordered_map<int, SIs> body;
  CollisionTable *CT;

  SITable(CollisionTable *_CT);
  ~SITable();
  SIs &get(Vertex *v);
};

struct SINode {
  const int uuid;
  const int time_start;
  const int time_end;
  Vertex *v;
  const int t;  // arrival time
  const int g;
  const int f;
  SINode *parent;

  SINode(const int uuid, const SI &si, Vertex *_v, int _t, int _g, int _f,
         SINode *_parent);
  bool operator==(const SINode &other) const;
};
using SINodes = std::vector<SINode *>;

struct SINodeHasher {
  uint operator()(const SINode &n) const;
};

Path sipp(const int i, Vertex *s_i, Vertex *g_i, DistTable *D,
          CollisionTable *CT, const Deadline *deadline = nullptr,
          const int f_upper_bound = INT_MAX);

std::ostream &operator<<(std::ostream &os, const SINode *n);
