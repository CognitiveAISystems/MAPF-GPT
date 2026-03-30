#include "../include/heuristic.hpp"

Heuristic::Heuristic(const Instance *_ins, DistTable *_D) : ins(_ins), D(_D) {}

int Heuristic::get(const Config &Q)
{
  auto cost = 0;
  for (size_t i = 0; i < ins->N; ++i) cost += D->get(i, Q[i]);
  return cost;
}
