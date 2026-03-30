#include "../include/planner.hpp"

#include <algorithm>
#include <iostream>

bool Planner::FLG_SWAP = true;
bool Planner::FLG_STAR = true;
bool Planner::FLG_MULTI_THREAD = true;
int Planner::SCATTER_MARGIN = 10;
int Planner::PIBT_NUM = 10;
bool Planner::FLG_REFINER = true;
int Planner::REFINER_NUM = 4;
bool Planner::FLG_SCATTER = true;
float Planner::RANDOM_INSERT_PROB1 = 0.1;
float Planner::RANDOM_INSERT_PROB2 = 0.01;
bool Planner::FLG_RANDOM_INSERT_INIT_NODE = false;
float Planner::RECURSIVE_RATE = 0.2;
double Planner::RECURSIVE_TIME_LIMIT = 1000;

std::string Planner::MSG;
int Planner::CHECKPOINTS_DURATION = 5000;
constexpr int CHECKPOINTS_NIL = -1;

constexpr auto TIME_ZERO = std::chrono::seconds(0);

Planner::Planner(const Instance *_ins, int _verbose, const Deadline *_deadline,
                 int _seed, int _depth, DistTable *_D)
    : ins(_ins),
      deadline(_deadline),
      seed(_seed),
      MT(std::mt19937(seed)),
      verbose(_verbose),
      depth(_depth),
      N(ins->N),
      V_size(ins->G->size()),
      D((_D == nullptr) ? new DistTable(ins) : _D),
      delete_dist_table_after_used(_D == nullptr),
      heuristic(new Heuristic(ins, D)),
      scatter(nullptr),
      seed_refiner(0),
      refiner_pool(),
      OPEN(),
      EXPLORED(),
      H_init(nullptr),
      H_goal(nullptr),
      search_iter(0),
      time_initial_solution(-1),
      cost_initial_solution(-1),
      checkpoints()
{
}

Planner::~Planner()
{
  if (heuristic != nullptr) delete heuristic;
  if (scatter != nullptr) delete scatter;
  for (auto &pibt : pibts) delete pibt;
  if (delete_dist_table_after_used) delete D;
}

Solution Planner::solve()
{
  info(1, verbose, deadline, "start search");
  update_checkpoints();

  // insert initial node
  H_init = create_highlevel_node(ins->starts, nullptr);
  OPEN.push_front(H_init);

  set_scatter();
  set_pibt();

  // search loop
  while (!OPEN.empty() && !is_expired(deadline)) {
    search_iter += 1;
    update_checkpoints();

    // check pooled procedures
    refiner_pool.remove_if([&](auto &proc) {
      if ((proc).wait_for(TIME_ZERO) != std::future_status::ready) return false;
      apply_new_solution(proc.get());
      ++seed_refiner;
      refiner_pool.emplace_back(std::async(std::launch::async,
                                           &Planner::get_refined_plan, this,
                                           backtrack(H_goal)));
      return true;
    });

    // do not pop here!
    auto H = OPEN.front();

    // random insert after initial solution found
    if (H_goal != nullptr && get_random_float(MT) < RANDOM_INSERT_PROB2) {
      H = FLG_RANDOM_INSERT_INIT_NODE
              ? H_init
              : OPEN[get_random_int(MT, 0, OPEN.size() - 1)];
    }

    // check lower bounds
    if (H_goal != nullptr && H->f >= H_goal->f) {
      OPEN.pop_front();
      continue;
    }

    // check goal condition
    if (H_goal == nullptr && is_same_config(H->C, ins->goals)) {
      time_initial_solution = elapsed_ms(deadline);
      cost_initial_solution = H->g;
      H_goal = H;
      info(1, verbose, deadline, "found initial solution, cost: ", H_goal->g);
      if (!FLG_STAR) break;  // finish search
      set_refiner();         // refining start
      continue;
    }

    // low level search
    auto L = H->get_next_lowlevel_node(MT);
    if (L == nullptr) {
      OPEN.pop_front();
      continue;
    }

    // create successors at the high-level search
    auto Q_to = Config(N, nullptr);
    auto res = set_new_config(H, L, Q_to);
    delete L;
    if (!res) continue;

    // check explored list
    auto iter = EXPLORED.find(Q_to);
    if (iter != EXPLORED.end()) {
      // known configuration
      rewrite(H, iter->second);

      if (get_random_float(MT) >= RANDOM_INSERT_PROB1) {
        OPEN.push_front(iter->second);  // usual
      } else {
        OPEN.push_front(H_init);  // sometimes
      }
    } else {
      // new one -> insert
      auto H_new = create_highlevel_node(Q_to, H);
      OPEN.push_front(H_new);
    }
  }

  // clear pooled operaitons
  bool is_optimal = OPEN.empty();
  for (auto &proc : refiner_pool) apply_new_solution(proc.get());
  if (is_optimal) OPEN.clear();

  // end processing
  update_checkpoints();
  logging();
  auto solution = backtrack(H_goal);        // obtain solution
  for (auto p : EXPLORED) delete p.second;  // memory management
  return solution;
}

HNode *Planner::create_highlevel_node(const Config &Q, HNode *parent)
{
  auto g_val =
      (parent == nullptr) ? 0 : parent->g + get_edge_cost(parent->C, Q);
  auto h_val = heuristic->get(Q);
  auto H_new = new HNode(Q, D, parent, g_val, h_val);
  EXPLORED[Q] = H_new;
  return H_new;
}

void Planner::apply_new_solution(const Solution &plan)
{
  if (plan.empty()) return;
  info(3, verbose, deadline, "incorporate new solution");

  // forcibly insert configuration
  HNode *H_from = EXPLORED[plan[0]];
  HNode *H_to = nullptr;
  for (auto t = 1; t < plan.size(); ++t) {
    auto &&Q = plan[t];
    auto iter = EXPLORED.find(Q);
    if (iter != EXPLORED.end()) {
      // known
      H_to = iter->second;
      rewrite(H_from, H_to);
    } else {
      // new
      auto g_val = H_from->g + get_edge_cost(H_from->C, Q);
      H_to = new HNode(Q, D, H_from, g_val, heuristic->get(Q));
      EXPLORED[Q] = H_to;
      OPEN.push_front(H_to);
    }
    H_from = H_to;
  }
}

Solution Planner::backtrack(HNode *H)
{
  std::vector<Config> plan;
  auto _H = H;
  while (_H != nullptr) {
    plan.push_back(_H->C);
    _H = _H->parent;
  }
  std::reverse(plan.begin(), plan.end());
  return plan;
}

bool Planner::set_new_config(HNode *H, LNode *L, Config &Q_to)
{
  // worker-id, time -> configuration
  auto Q_cands = std::vector<Config>(PIBT_NUM, Config(N, nullptr));
  auto f_vals = std::vector<int>(PIBT_NUM, INT_MAX);

  // parallel
  auto worker = [&](int k) {
    // set constraints
    for (auto d = 0; d < L->depth; ++d) Q_cands[k][L->who[d]] = L->where[d];
    // PIBT
    auto res = pibts[k]->set_new_config(H->C, Q_cands[k], H->order);
    if (res)
      f_vals[k] = get_edge_cost(H->C, Q_cands[k]) + heuristic->get(Q_cands[k]);
  };
  if (FLG_MULTI_THREAD && PIBT_NUM > 1) {
    auto threads = std::vector<std::thread>();
    for (auto k = 0; k < PIBT_NUM; ++k) threads.emplace_back(worker, k);
    for (auto &th : threads) th.join();
  } else {
    for (auto k = 0; k < PIBT_NUM; ++k) worker(k);
  }

  // obtain the best score
  auto min_f_val = INT_MAX;
  auto min_f_val_idx = -1;
  for (auto k = 0; k < PIBT_NUM; ++k) {
    if (f_vals[k] < min_f_val) {
      min_f_val = f_vals[k];
      min_f_val_idx = k;
    }
  }

  if (min_f_val < INT_MAX) {
    auto &Q_win = Q_cands[min_f_val_idx];
    std::copy(Q_win.begin(), Q_win.end(), Q_to.begin());
    return true;
  } else {
    return false;
  }
}

void Planner::rewrite(HNode *H_from, HNode *H_to)
{
  // update neighbors
  H_from->neighbor.insert(H_to);

  // Dijkstra
  std::queue<HNode *> Q({H_from});  // queue is sufficient
  while (!Q.empty()) {
    auto n_from = Q.front();
    Q.pop();
    for (auto n_to : n_from->neighbor) {
      auto g_val = n_from->g + get_edge_cost(n_from->C, n_to->C);
      if (g_val < n_to->g) {
        if (n_to == H_goal)
          info(2, verbose, deadline, "cost update: ", H_goal->g, " -> ", g_val);
        n_to->g = g_val;
        n_to->f = n_to->g + n_to->h;
        n_to->parent = n_from;
        Q.push(n_to);
        if (H_goal != nullptr && n_to->f < H_goal->f) OPEN.push_front(n_to);
      }
    }
  }
}

int Planner::get_edge_cost(const Config &C1, const Config &C2)
{
  auto cost = 0;
  for (uint i = 0; i < N; ++i) {
    if (C1[i] != ins->goals[i] || C2[i] != ins->goals[i]) {
      cost += 1;
    }
  }
  return cost;
}

void Planner::set_scatter()
{
  if (!FLG_SCATTER) return;
  info(1, verbose, deadline, "start computing SUO");
  auto scatter_deadline =
      Deadline(deadline == nullptr
                   ? INT_MAX
                   : (deadline->time_limit_ms - elapsed_ms(deadline)) / 2);
  auto margin = SCATTER_MARGIN < 0 ? get_random_int(MT, 0, 30) : SCATTER_MARGIN;
  scatter = new Scatter(ins, D, &scatter_deadline, 3, verbose - 4, margin);
  scatter->construct();
  info(1, verbose, deadline, "finish computing SUO",
       ", collision count: ", scatter->CT.collision_cnt,
       ", scatter margin: ", scatter->cost_margin,
       ", sum_of_path_length: ", scatter->sum_of_path_length);
}

void Planner::set_pibt()
{
  for (auto k = 0; k < PIBT_NUM; ++k) {
    pibts.emplace_back(new PIBT(ins, D, k + seed, FLG_SWAP, scatter));
  }
}

void Planner::set_refiner()
{
  if (!FLG_REFINER) return;
  if (!FLG_MULTI_THREAD) return;
  auto plan = backtrack(H_goal);
  info(2, verbose, deadline, "invoke refiners");
  for (auto k = 0; k < REFINER_NUM; ++k) {
    ++seed_refiner;
    refiner_pool.emplace_back(
        std::async(std::launch::async, &Planner::get_refined_plan, this, plan));
  }
}

Solution Planner::get_refined_plan(const Solution &plan)
{
  auto MT_internal = std::mt19937(seed_refiner);
  if (depth < 1 && plan.size() > 3 &&
      get_random_float(MT_internal) < RECURSIVE_RATE) {
    // recursive LaCAM
    auto ins_tmp =
        Instance(ins->G, plan[get_random_int(MT_internal, 1, plan.size() - 2)],
                 ins->goals, N);
    auto deadline_tmp = Deadline(std::min(
        RECURSIVE_TIME_LIMIT,
        deadline == nullptr ? INT_MAX
                            : deadline->time_limit_ms - elapsed_ms(deadline)));
    auto planner_tmp =
        Planner(&ins_tmp, 0, &deadline_tmp, seed_refiner, depth + 1, D);
    info(4, verbose, deadline, "refiner-", planner_tmp.seed,
         "\tactivated (recursive LaCAM)");
    auto res = planner_tmp.solve();
    info(4, verbose, deadline, "refiner-", planner_tmp.seed,
         "\tcompleted (recursive LaCAM)");
    return res;
  } else if (RECURSIVE_RATE < 1.0) {
    // iterative refinement
    return refine(ins, deadline, plan, D, seed_refiner, verbose - 4);
  } else {
    return Solution();
  }
}

void Planner::update_checkpoints()
{
  const auto time = elapsed_ms(deadline);
  while (time >= checkpoints.size() * CHECKPOINTS_DURATION) {
    checkpoints.push_back(H_goal != nullptr ? H_goal->f : CHECKPOINTS_NIL);
  }
}

void Planner::logging()
{
  if (depth > 0) return;
  MSG += "checkpoints=";
  for (auto &k : checkpoints) MSG += std::to_string(k) + ",";
  MSG +=
      "\ncomp_time_initial_solution=" + std::to_string(time_initial_solution);
  MSG += "\ncost_initial_solution=" + std::to_string(cost_initial_solution);
  MSG += "\nsearch_iteration=" + std::to_string(search_iter);
  MSG += "\nnum_high_level_node=" + std::to_string(HNode::COUNT);
  MSG += "\nnum_low_level_node=" + std::to_string(LNode::COUNT);

  if (H_goal != nullptr && OPEN.empty()) {
    info(1, verbose, deadline, "solved optimally, cost:", H_goal->g);
  } else if (H_goal != nullptr) {
    info(1, verbose, deadline, "solved sub-optimally, cost:", H_goal->g);
  } else if (OPEN.empty()) {
    info(1, verbose, deadline, "no solution");
  } else {
    info(1, verbose, deadline, "timeout");
  }
  info(1, verbose, deadline, "search iteration:", search_iter,
       "\texplored:", EXPLORED.size());
}
