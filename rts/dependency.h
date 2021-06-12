
#pragma once

#include <map>
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

public:
  std::map <std::pair<int64_t, int64_t>, dependency_t> table;

  dependency_table() : table() {};

  ~dependency_table() {}

  void update(int64_t inst_from, int64_t inst_to, dependency_t dep);

  void print() const;

};

