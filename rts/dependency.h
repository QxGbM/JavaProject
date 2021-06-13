
#pragma once

#include <map>
#include <set>
#include <vector>
#include <cstdint>

enum class dependency_t : int { 
  no = 0,
  flow = 1,
  anti = 2,
  flow_anti = 3,
  out = 4,
  flow_out = 5,
  anti_out = 6,
  flow_anti_out = 7
};

dependency_t operator+ (dependency_t dep1, dependency_t dep2);

bool operator> (dependency_t dep1, dependency_t dep2);

class dependency_table {
private:
  std::set <std::pair<int64_t, int64_t>> table;
  std::multimap <int64_t, int64_t> table_f;
  std::multimap <int64_t, int64_t> table_t;
  std::map <std::pair<int64_t, int64_t>, dependency_t> type;

public:

  dependency_table() : table(), table_f(), table_t(), type() {};

  ~dependency_table() {}

  void update(int64_t inst_from, int64_t inst_to, dependency_t dep);

  dependency_t lookup(int64_t inst_from, int64_t inst_to) const;

  void getDepFrom(int64_t from, std::vector<int64_t>& lis) const;

  void getDepTo(int64_t to, std::vector<int64_t>& lis) const;

  size_t countDepFrom(int64_t from) const;

  size_t countDepTo(int64_t to) const;

  void clear();

  void print() const;

};

