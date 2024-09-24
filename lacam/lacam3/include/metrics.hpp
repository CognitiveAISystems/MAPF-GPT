/*
 * solution evaluation metrics
 */

#pragma once

#include "dist_table.hpp"
#include "instance.hpp"
#include "utils.hpp"

int get_makespan(const Solution &solution);
int get_makespan_paths(const std::vector<Path> &solution);

int get_path_cost(const Solution &solution, int i);  // single-agent path cost
int get_path_cost(const Path &path);
int get_sum_of_costs(const Solution &solution);
int get_sum_of_costs_paths(const std::vector<Path> &solution);

int get_path_loss(const Path &path);
int get_sum_of_loss(const Solution &solution);
int get_sum_of_loss(const Solution &solution, std::vector<int> &agents_subset);
int get_sum_of_loss_paths(const std::vector<Path> &solution);

int get_makespan_lower_bound(const Instance &ins, DistTable &D);
int get_sum_of_costs_lower_bound(const Instance &ins, DistTable &D);
