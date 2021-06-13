
#include <dependency.h>
#include <iostream>
#include <cstdlib>
#include <memory>

dependency_t operator+ (dependency_t dep1, dependency_t dep2) { return (dependency_t) ((int) dep1 | (int) dep2); }

bool operator> (dependency_t dep1, dependency_t dep2) { return (int) dep1 > (int) dep2; }

void dependency_table::update(int64_t inst_from, int64_t inst_to, dependency_t dep) {
  if (dep > dependency_t::no) {
    auto ret = table.insert(std::make_pair(inst_from, inst_to));
    if (ret.second) {
      table_f.insert(std::make_pair(inst_from, inst_to));
      table_t.insert(std::make_pair(inst_to, inst_from));
      type.insert(std::make_pair(std::make_pair(inst_from, inst_to), dep));
    }
    else
      type.find(std::make_pair(inst_from, inst_to))->second = dep;
  }
  else {
    auto i = table.find(std::make_pair(inst_from, inst_to));
    if (i != table.end()) {
      table.erase(i);
      auto ret = table_f.equal_range(inst_from);
      for (auto i = ret.first; i != ret.second; i++)
        if (i->second == inst_to)
          table_f.erase(i);
      ret = table_t.equal_range(inst_to);
      for (auto i = ret.first; i != ret.second; i++)
        if (i->second == inst_from)
          table_t.erase(i);
      type.erase(type.find(std::make_pair(inst_from, inst_to)));
    }
  }
}

dependency_t dependency_table::lookup(int64_t inst_from, int64_t inst_to) const {
  return 
    table.find(std::make_pair(inst_from, inst_to)) == table.end() ? dependency_t::no :
    type.find(std::make_pair(inst_from, inst_to))->second;
}

void dependency_table::getDepFrom(int64_t from, std::vector<int64_t>& lis) const {
  lis.clear();
  auto ret = table_f.equal_range(from);
  for (auto i = ret.first; i != ret.second; i++)
    lis.emplace_back(i->second);
}

void dependency_table::getDepTo(int64_t to, std::vector<int64_t>& lis) const {
  lis.clear();
  auto ret = table_t.equal_range(to);
  for (auto i = ret.first; i != ret.second; i++)
    lis.emplace_back(i->second);
}

size_t dependency_table::countDepFrom(int64_t from) const {
  return table_f.count(from);
}

size_t dependency_table::countDepTo(int64_t to) const {
  return table_t.count(to);
}

void dependency_table::clear() {
  table.clear();
  table_f.clear();
  table_t.clear();
}

void dependency_table::print() const {
  for (auto& i : table)
  { std::cout << i.first << " " << i.second << std::endl; }
}

