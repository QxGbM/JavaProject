#ifndef _DEV_HIERARCHICAL_OPS_DAG_CUH
#define _DEV_HIERARCHICAL_OPS_DAG_CUH

#include <pspl.cuh>

class dependency_linked_list
{
private:

  int to;
  dependency_t dependency;
  dependency_linked_list * next;

public:

  __host__ dependency_linked_list (const int to_in, const dependency_t dependency_in)
  {
    to = to_in;
    dependency = dependency_in;
    next = nullptr;
  }

  __host__ ~dependency_linked_list ()
  {
    delete next;
  }

  __host__ void insertDependency (const int to_in, const dependency_t dependency_in)
  {
    for (dependency_linked_list * ptr = this; ptr != nullptr; ptr = ptr -> next)
    {
      if (ptr -> next == nullptr) 
      { ptr -> next = new dependency_linked_list(to_in, dependency_in); return; }
    }
  }

  __host__ dependency_t lookupDependency (const int to_in) const
  {
    for (const dependency_linked_list * ptr = this; ptr != nullptr; ptr = ptr -> next)
	  { 
      if (ptr -> to == to_in) 
      { return ptr -> dependency; }
      else if (ptr -> to > to_in)
      { return no_dep; }
    }
    return no_dep;
  }

  __host__ int length () const
  {
	  int l = 0;
	  for (const dependency_linked_list * ptr = this; ptr != nullptr; ptr = ptr -> next)
	  { l++; }
    return l;
  }

  __host__ void print () const
  {
    for (const dependency_linked_list * ptr = this; ptr != nullptr; ptr = ptr -> next)
	  { printf("%d ", ptr -> to); }
    printf("\n");
  }
};

class h_ops_dag 
{
private:

  int length;
  unsigned long long int fops;
  h_ops_tree * ops_list;
  dependency_linked_list ** deps_graph;

public:

  __host__ h_ops_dag (const h_ops_tree * ops) 
  {
    ops_list = ops -> flatten();
    //ops_list->print();
    fops = ops_list -> getFops_All();
    length = ops_list -> length();
    deps_graph = new dependency_linked_list * [length];

    memset(deps_graph, 0, length * sizeof(dependency_linked_list *));

    int i = 1;
    for (h_ops_tree * to = ops_list -> getNext(); i < length; to = to -> getNext(), i++)
    {
      int j = 0;
      for (h_ops_tree * from = ops_list; j < i; from = from -> getNext(), j++)
      {
        dependency_t dep = to -> checkDependencyFrom(from);
        if (dep > no_dep)
        {
          if (deps_graph[j] == nullptr)
          { deps_graph[j] = new dependency_linked_list(i, dep); }
          else
          { deps_graph[j] -> insertDependency(i, dep); }
        }
      }
    }

  }

  __host__ ~h_ops_dag ()
  {
    for (int i = 0; i < length; i++)
    { delete deps_graph[i]; }
    delete[] deps_graph;

    delete ops_list;
  }

  __host__ int getLength () const
  {
    return length;
  }

  __host__ h_ops * getOp (const int i) const
  {
    int c = 0;
    for (h_ops_tree * op = ops_list; op != nullptr; op = op -> getNext())
    { if (c == i) { return op; } c++; }
    return nullptr;
  }

  __host__ dependency_t getDep (const int from, const int to) const
  {
    if (deps_graph[from] == nullptr) 
    { return no_dep; }
    else
    { return deps_graph[from] -> lookupDependency(to); }
  }

  __host__ inline int getDepCount_from (const int from) const
  { return (deps_graph[from] == nullptr) ? 0 : deps_graph[from] -> length(); }

  __host__ int getDepCount_to (const int to) const
  {
    int sum = 0;
    for (int i = 0; i < to; i++)
    { if (getDep(i, to) > no_dep) { sum++; } }
    return sum;
  }

  __host__ unsigned long long int getFops () const
  {
    return fops;
  }

  __host__ void print() const
  {
    h_ops_tree * to = ops_list;
    for (int i = 0; i < length; i++, to = to -> getNext())
    {
      printf("Inst %d: ", i);
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
      printf("\n");
    }
    printf("Total fp-ops: %llu.\n\n", fops);
  }
  
};


#endif