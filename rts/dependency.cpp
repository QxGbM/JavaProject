
#include <dependency.h>
#include <iostream>
#include <cstdlib>
#include <memory>

dependency_t operator+ (dependency_t dep1, dependency_t dep2) { return (dependency_t) ((int) dep1 | (int) dep2); }

bool operator> (dependency_t dep1, dependency_t dep2) { return (int) dep1 > (int) dep2; }

void dependency_table::update(int64_t inst_from, int64_t inst_to, dependency_t dep) {
  auto i = table.find(std::make_pair(inst_from, inst_to));
  if (i == table.end())
    table.insert(std::make_pair(std::make_pair(inst_from, inst_to), dep));
  else
    i->second = dep;
}

void dependency_table::print() const {
  for (auto& i : table)
  { std::cout << i.first.first << " " << i.first.second << std::endl; }
}

