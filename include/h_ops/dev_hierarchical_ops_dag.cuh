
#pragma once
#ifndef _DEV_HIERARCHICAL_OPS_DAG_CUH
#define _DEV_HIERARCHICAL_OPS_DAG_CUH

#include <pspl.cuh>

class dependency_linked_list
{
private:

  int inst;
  dependency_t dependency;
  dependency_linked_list * next;

public:

  dependency_linked_list (const int inst_in, const dependency_t dependency_in, dependency_linked_list * next_in = nullptr)
  {
    inst = inst_in;
    dependency = dependency_in;
    next = next_in;
  }

  ~dependency_linked_list ()
  { delete next; }

  int getInst () const
  { return inst; }

  dependency_t getDep () const
  { return dependency; }

  dependency_linked_list * getNext () const
  { return next; }

  dependency_t lookupDependency (const int inst_in) const
  {
    for (const dependency_linked_list * ptr = this; ptr != nullptr; ptr = ptr -> next)
    { 
      if (ptr -> inst == inst_in) { return ptr -> dependency; }
      else if (ptr -> inst < inst_in) { return no_dep; }
    }
    return no_dep;
  }

  int length () const
  {
    int l = 0;
    for (const dependency_linked_list * ptr = this; ptr != nullptr; ptr = ptr -> next) { l++; }
    return l;
  }

  void print () const
  {
    for (const dependency_linked_list * ptr = this; ptr != nullptr; ptr = ptr -> next) { printf("%d ", ptr -> inst); }
    printf("\n");
  }

};

class h_ops_dag 
{
private:

  int length;
  long long int * flops;
  long long int * flops_trim;
  h_ops_tree * ops_list;
  dependency_linked_list ** deps_graph_from;
  dependency_linked_list ** deps_graph_to;

public:

  h_ops_dag (const h_ops_tree * ops, const int start_index = 0, const int length_max = 0) 
  {
    ops_list = ops -> flatten(start_index, length_max);
    length = ops_list -> length();

    flops = new long long int [length];
    flops_trim = new long long int [length];

    deps_graph_from = new dependency_linked_list * [length];
    deps_graph_to = new dependency_linked_list * [length];

#pragma omp parallel for
    for (int i = 0; i < length; i++)
    {
      dependency_linked_list * list = nullptr;
      h_ops_tree * from = ops_list -> getChild(i);
      for (int j = i + 1; j < length; j++)
      {
        dependency_t dep = ops_list -> getChild(j) -> checkDependencyFrom(from); 
        if (dep > no_dep)
        { list = new dependency_linked_list(j, dep, list); }
      }
      flops[i] = from -> getFlops(&flops_trim[i]);
      deps_graph_from[i] = list;
    }

#pragma omp parallel for
    for (int i = 0; i < length; i++)
    {
      dependency_linked_list * list = nullptr;
      for (int j = 0; j < i; j++)
      {
        dependency_t dep = deps_graph_from[j] -> lookupDependency(i);
        if (dep > no_dep)
        { list = new dependency_linked_list(j, dep, list); }
      }
      deps_graph_to[i] = list;
    }

  }

  ~h_ops_dag ()
  {
    for (int i = 0; i < length; i++)
    { delete deps_graph_from[i]; delete deps_graph_to[i]; }

    delete[] deps_graph_from;
    delete[] deps_graph_to;
    delete ops_list;
  }

  int getLength () const
  { return length; }

  h_ops * getOp (const int index) const
  { return ops_list -> getChild(index); }

  dependency_t getDep (const int from, const int to) const
  {
    if (deps_graph_from[from] == nullptr) 
    { return no_dep; }
    else
    { return deps_graph_from[from] -> lookupDependency(to); }
  }

  dependency_linked_list * getDepList_From (const int from) const
  {
    if (from >= 0 && from < length)
    { return deps_graph_from[from]; }
    else
    { return nullptr; }
  }

  dependency_linked_list * getDepList_To (const int to) const
  {
    if (to >= 0 && to < length)
    { return deps_graph_to[to]; }
    else
    { return nullptr; }
  }

  int getDepCount_From (const int from) const
  {
    if (from >= 0 && from < length)
    { return deps_graph_from[from] -> length(); }
    else
    { return 0; }
  }

  int * getDepCountList_To () const
  {
    int * deps = new int [length];

#pragma omp parallel for
    for (int i = 0; i < length; i++)
    { deps[i] = deps_graph_to[i] -> length(); }

    return deps;
  }

  long long int getFlops (const int index = -1) const
  { 
    if (index >= 0 && index < length) 
    { return flops[index]; }
    else
    {
      long long int accum = 0;
#pragma omp parallel for reduction(+:accum)
      for (int i = 0; i < length; i++)
      { accum += flops[i]; }
      return accum;
    }
  }

  long long int getFlops_Trim (const int index = -1) const
  { 
    if (index >= 0 && index < length) 
    { return flops_trim[index]; }
    else
    {
      long long int accum = 0;
#pragma omp parallel for reduction(+:accum)
      for (int i = 0; i < length; i++)
      { accum += flops_trim[i]; }
      return accum;
    }
  }

  void print() const
  {
    for (int i = 0; i < length; i++)
    {
      printf("Inst %d: ", i);
      h_ops_tree * to = ops_list -> getChild(i);
      to -> h_ops::print();
      for (int j = 0; j < i; j++)
      {
        dependency_t dep = getDep(j, i);
        if (dep > no_dep)
        { 
          printf("[%d: ", j);
          switch (dep)
          {
          case no_dep: break;
          case flow_dep: printf("FD"); break;
          case anti_dep: printf("AD"); break;
          case flow_anti_dep: printf("FAD"); break;
          case output_dep: printf("OD"); break;
          case flow_output_dep: printf("FOD"); break;
          case anti_output_dep: printf("AOD"); break;
          case flow_anti_output_dep: printf("FAOD"); break;
          }
          printf("] ");
        }
      }
      printf("Flops: %lld \n", flops[i]);
    }
    printf("Total Flops: %lld.\n\n", getFlops());
  }
  
};


#endif