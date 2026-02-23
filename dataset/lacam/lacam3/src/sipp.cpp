#include "../include/sipp.hpp"

SITable::SITable(CollisionTable *_CT) : CT(_CT) {}

SITable::~SITable() {}

SIs &SITable::get(Vertex *v)
{
  auto &b_v = body[v->id];
  if (!b_v.empty()) return b_v;
  auto &entry = CT->body[v->id];
  auto &entry_last = CT->body_last[v->id];
  auto t_last = entry_last.empty()
                    ? INT_MAX
                    : *std::min_element(entry_last.begin(), entry_last.end());

  // insert safe interval
  auto time_start = 0;
  for (auto t = 0; t < entry.size(); ++t) {
    if (entry[t].empty()) continue;
    auto time_end = t - 1;
    if (time_start <= time_end) {
      b_v.push_back(std::make_pair(time_start, time_end));
    }
    time_start = t + 1;
    if (t_last == t) break;
  }
  // add last safe interval
  if (t_last == INT_MAX) {
    b_v.push_back(std::make_pair(time_start, INT_MAX - 1));
  }
  return b_v;
}

SINode::SINode(const int _uuid, const SI &si, Vertex *_v, int _t, int _g,
               int _f, SINode *_parent)
    : uuid(_uuid),
      time_start(si.first),
      time_end(si.second),
      v(_v),
      t(_t),
      g(_g),
      f(_f),
      parent(_parent)
{
}

bool SINode::operator==(const SINode &other) const
{
  return (other.v->id == v->id && other.time_start == time_start &&
          other.time_end == time_end);
}

uint SINodeHasher::operator()(const SINode &n) const
{
  uint hash = n.v->id;
  hash ^= n.time_start + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  hash ^= n.time_end + 0x9e3779b9 + (hash << 6) + (hash >> 2);
  return hash;
}

// minimizing path-loss - not cost!
Path sipp(const int i, Vertex *s_i, Vertex *g_i, DistTable *D,
          CollisionTable *CT, const Deadline *deadline, const int f_upper_bound)
{
  auto solution_path = Path();
  auto ST = SITable(CT);  // safe interval table

  // setup goal
  auto &intervals_goal = ST.get(g_i);
  if (intervals_goal.empty()) return solution_path;
  const auto t_goal_after = intervals_goal.back().first - 1;

  // setup OPEN lists
  auto cmpNodes = [&](SINode *a, SINode *b) {
    if (a->f != b->f) return a->f > b->f;
    if (a->g != b->g) return a->g < b->g;
    if (a->time_start != b->time_start) return a->time_start > b->time_start;
    return a->uuid < b->uuid;
  };

  int node_id = 0;
  auto OPEN =
      std::priority_queue<SINode *, SINodes, decltype(cmpNodes)>(cmpNodes);
  std::unordered_map<SINode, SINode *, SINodeHasher> EXPLORED;
  OPEN.push(new SINode(++node_id, ST.get(s_i)[0], s_i, 0, 0, D->get(i, s_i),
                       nullptr));

  // main loop
  while (!OPEN.empty() && !is_expired(deadline)) {
    auto n = OPEN.top();
    OPEN.pop();

    // check known node
    auto itr_e = EXPLORED.find(*n);
    if (itr_e != EXPLORED.end() && itr_e->second->g <= n->g) {
      delete n;
      continue;
    }
    EXPLORED[*n] = n;

    // goal check
    if (n->v == g_i && n->t > t_goal_after) {
      // backtrack
      auto t = n->t;
      while (t >= 0) {
        solution_path.push_back(n->v);
        if (t == n->t) n = n->parent;
        --t;
      }
      std::reverse(solution_path.begin(), solution_path.end());
      break;
    }

    // expand neighbors
    for (auto &u : n->v->neighbor) {
      for (auto &si : ST.get(u)) {
        // invalid transition
        if (si.first > n->time_end + 1) break;
        if (si.second <= n->time_start) continue;

        // check existence of t
        auto t_earliest = INT_MAX;
        if (n->v != g_i) {
          for (auto t = std::max(n->t, si.first - 1);
               t <= std::min(n->time_end, si.second - 1); ++t) {
            if (CT->getCollisionCost(n->v, u, t) == 0) {
              t_earliest = t + 1;
              break;
            }
          }
        } else {
          // for goal node -> reverse
          for (auto t = std::min(n->time_end, si.second - 1);
               t >= std::max(n->t, si.first - 1); --t) {
            if (CT->getCollisionCost(n->v, u, t) == 0) {
              t_earliest = t + 1;
              break;
            }
          }
        }
        if (t_earliest >= INT_MAX) continue;

        // valid neighbor
        auto g_val = n->g + (n->v != g_i ? t_earliest - n->t : 1);
        auto f_val = g_val + D->get(i, u);
        auto n_new = new SINode(++node_id, si, u, t_earliest, g_val, f_val, n);

        auto itr = EXPLORED.find(*n_new);
        if (f_val > f_upper_bound ||
            (itr != EXPLORED.end() && g_val >= itr->second->g)) {
          delete n_new;
        } else {
          OPEN.push(n_new);
        }
      }
    }
  }

  // memory management
  while (!OPEN.empty()) {
    delete OPEN.top();
    OPEN.pop();
  }
  for (auto iter : EXPLORED) delete iter.second;
  return solution_path;
}

std::ostream &operator<<(std::ostream &os, const SINode *n)
{
  os << "f=" << std::setw(4) << n->f << ", v=" << std::setw(6) << n->v
     << ", t=" << std::setw(4) << n->t << ", si: [" << std::setw(4)
     << n->time_start << ", " << std::setw(4)
     << ((n->time_end < INT_MAX - 1) ? std::to_string(n->time_end) : "inf")
     << "]";
  return os;
}
