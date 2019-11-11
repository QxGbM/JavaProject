
#pragma once
#ifndef _DEPENDENCY_CUH
#define _DEPENDENCY_CUH

#include <definitions.cuh>

enum class dependency_t { 
  no_dep = 0,
  flow_dep = 1,
  anti_dep = 2,
  flow_anti_dep = 3,
  output_dep = 4,
  flow_output_dep = 5,
  anti_output_dep = 6,
  flow_anti_output_dep = 7
};

dependency_t operator+ (dependency_t dep1, dependency_t dep2);

bool operator> (dependency_t dep1, dependency_t dep2);

class dependency_linked_list
{
private:

  int inst;
  dependency_t dependency;
  dependency_linked_list * next;

public:

  dependency_linked_list (const int inst_in, const dependency_t dependency_in, dependency_linked_list * next_in = nullptr);

  ~dependency_linked_list ();

  int getInst () const;

  dependency_t getDep () const;

  dependency_linked_list * getNext () const;

  dependency_t lookupDependency (const int inst_in) const;

  void addDependency (const int inst_in, const dependency_t dependency_in);

  void updateDependency (const int inst_in, const dependency_t dependency_in);

  void addInstOffset (const int offset);

  int length () const;

  void print () const;

};


class dependency_table
{
private:

  int size;
  dependency_linked_list ** from;
  dependency_linked_list ** to;

public:

  dependency_table (int size_in);

  ~dependency_table ();

  void resize (int size_new);

  void addDependency (const int inst_from, const int inst_to, dependency_t dep);

  void updateDependency (const int inst_from, const int inst_to, dependency_t dep);

  void concatTable (dependency_table * table);

  void print() const;

};

#endif