/*
 * Implementation of LaCAM*
 *
 * references:
 * LaCAM: Search-Based Algorithm for Quick Multi-Agent Pathfinding.
 * Keisuke Okumura.
 * Proc. AAAI Conf. on Artificial Intelligence (AAAI). 2023.
 *
 * Improving LaCAM for Scalable Eventually Optimal Multi-Agent Pathfinding.
 * Keisuke Okumura.
 * Proc. Int. Joint Conf. on Artificial Intelligence (IJCAI). 2023.
 *
 * Engineering LaCAM*: Towards Real-Time, Large-Scale, and Near-Optimal
 * Multi-Agent Pathfinding. Keisuke Okumura. Proc. Int. Conf. on Autonomous
 * Agents and Multiagent Systems. 2024.
 */
#pragma once

#include "dist_table.hpp"
#include "graph.hpp"
#include "heuristic.hpp"
#include "hnode.hpp"
#include "instance.hpp"
#include "pibt.hpp"
#include "refiner.hpp"
#include "scatter.hpp"
#include "translator.hpp"
#include "utils.hpp"

struct Planner {
  const Instance *ins;
  const Deadline *deadline;
  const int seed;
  std::mt19937 MT;
  const int verbose;
  const int depth;

  // solver utils
  const int N;  // number of agents
  const int V_size;
  DistTable *D;
  bool delete_dist_table_after_used;

  // heuristic
  Heuristic *heuristic;

  // scatter (SUO)
  Scatter *scatter;

  // configuration generator
  std::vector<PIBT *> pibts;

  // for refiner
  int seed_refiner;
  std::list<std::future<Solution>> refiner_pool;

  // for search utils
  std::deque<HNode *> OPEN;
  std::unordered_map<Config, HNode *, ConfigHasher> EXPLORED;
  HNode *H_init;  // start node
  HNode *H_goal;  // goal node

  // parameters
  static bool FLG_SWAP;  // whether to use swap technique in PIBT
  static bool
      FLG_STAR;  // whether to refine solutions after initial solution discovery
  static bool FLG_MULTI_THREAD;
  static int SCATTER_MARGIN;  // used in SUO
  static int PIBT_NUM;  // number of PIBT run, i.e., Monte-Carlo configuration
                        // generator
  static bool FLG_REFINER;  // whether to use refiners
  static int REFINER_NUM;   // number of refiners
  static bool
      FLG_SCATTER;  // whether to use space utilization optimization (SUO)
  static float RANDOM_INSERT_PROB1;  // inserting the start node
  static float RANDOM_INSERT_PROB2;  // inserting a node after finding the goal
  static bool FLG_RANDOM_INSERT_INIT_NODE;
  static float RECURSIVE_RATE;
  static double RECURSIVE_TIME_LIMIT;

  // for logging
  static int CHECKPOINTS_DURATION;
  static std::string MSG;

  int search_iter;
  int time_initial_solution;
  int cost_initial_solution;
  std::vector<int> checkpoints;

  Planner(const Instance *_ins, int _verbose = 0,
          const Deadline *_deadline = nullptr, int _seed = 0,
          int _depth = 0,          // used in recursive LaCAM
          DistTable *_D = nullptr  // used in recursive LaCAM
  );
  ~Planner();
  Solution solve();
  bool set_new_config(HNode *S, LNode *M, Config &Q_to);
  HNode *create_highlevel_node(const Config &Q, HNode *parent);
  void rewrite(HNode *H_from, HNode *H_to);
  int get_edge_cost(const Config &C1, const Config &C2);
  Solution backtrack(HNode *H);
  void apply_new_solution(const Solution &plan);
  void set_scatter();
  void set_pibt();
  void set_refiner();
  Solution get_refined_plan(const Solution &plan_origin);
  void update_checkpoints();
  void logging();
};
