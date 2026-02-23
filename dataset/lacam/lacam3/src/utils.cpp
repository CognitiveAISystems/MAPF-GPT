#include "../include/utils.hpp"

void info(const int level, const int verbose) { std::cout << std::endl; }

Deadline::Deadline(double _time_limit_ms)
    : t_s(Time::now()), time_limit_ms(_time_limit_ms)
{
}

double Deadline::elapsed_ms() const
{
  return std::chrono::duration_cast<std::chrono::milliseconds>(Time::now() -
                                                               t_s)
      .count();
}

double Deadline::elapsed_ns() const
{
  return std::chrono::duration_cast<std::chrono::nanoseconds>(Time::now() - t_s)
      .count();
}

double elapsed_ms(const Deadline *deadline)
{
  if (deadline == nullptr) return 0;
  return deadline->elapsed_ms();
}

double elapsed_ns(const Deadline *deadline)
{
  if (deadline == nullptr) return 0;
  return deadline->elapsed_ns();
}

bool is_expired(const Deadline *deadline)
{
  if (deadline == nullptr) return false;
  return deadline->elapsed_ms() > deadline->time_limit_ms;
}

float get_random_float(std::mt19937 &MT, float from, float to)
{
  return std::uniform_real_distribution<float>(from, to)(MT);
}

float get_random_float(std::mt19937 *MT, float from, float to)
{
  return get_random_float(*MT, from, to);
}

int get_random_int(std::mt19937 &MT, int from, int to)
{
  return std::uniform_int_distribution<int>(from, to)(MT);
}

int get_random_int(std::mt19937 *MT, int from, int to)
{
  return get_random_int(*MT, from, to);
}

std::ostream &operator<<(std::ostream &os, const std::vector<int> &arr)
{
  for (auto ele : arr) os << ele << ",";
  return os;
}

std::ostream &operator<<(std::ostream &os, const std::set<int> &arr)
{
  for (auto ele : arr) os << ele << ",";
  return os;
}
