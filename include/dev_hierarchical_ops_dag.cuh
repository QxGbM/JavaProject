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
    if (next == nullptr) 
    { next = new dependency_linked_list(to_in, dependency_in); }
    else
    { next -> insertDependency(to_in, dependency_in); }
  }

  __host__ dependency_t lookupDependency (const int to_in)
  {
    if (to_in == to)
    { return dependency; }
    else if (to_in < to || next == nullptr)
    { return no_dep; }
    else
    { return next -> lookupDependency(to_in); }
  }

  __host__ int length () const
  {
    return 1 + ((next == nullptr) ? 0 : next -> length());
  }

  __host__ void flatten (int * deps) const
  {
    deps[0] = to;
    if (next != nullptr) { next -> flatten(&deps[1]); }
  }

  __host__ void print () const
  {
    printf("%d ", to);
    if (next == nullptr)
    { printf("\n"); }
    else
    { next -> print(); }
  }
};

class h_ops_dag 
{
private:

  int length;
  h_ops * ops_list;
  dependency_linked_list ** deps_graph;
  unsigned long long int fops;

public:

  __host__ h_ops_dag (const h_ops_tree * ops) 
  {
    length = ops -> length();

    ops_list = new h_ops [length];
    deps_graph = new dependency_linked_list * [length];

    ops -> flatten(ops_list);
    memset(deps_graph, 0, length * sizeof(dependency_linked_list *));

    for (int i = 0; i < length; i++)
    {
      for (int j = 0; j < i; j++)
      {
        dependency_t dep = ops_list[i].checkDependencyFrom(&ops_list[j]);
        if (dep > no_dep)
        {
          if (deps_graph[j] == nullptr)
          { deps_graph[j] = new dependency_linked_list(i, dep); }
          else
          { deps_graph[j] -> insertDependency(i, dep); }
        }
      }
    }

    fops = ops -> getFops_All();
  }

  __host__ ~h_ops_dag ()
  {
    for (int i = 0; i < length; i++)
    { delete deps_graph[i]; }
    delete[] ops_list;
    delete[] deps_graph;

    printf("-- DAG destroyed. --\n\n");
  }

  __host__ int getLength () const
  {
    return length;
  }

  __host__ h_ops * getOps (const int i) const
  {
    return &ops_list[i];
  }

  __host__ dependency_t getDep (const int from, const int to) const
  {
    if (deps_graph[from] == nullptr) 
    { return no_dep; }
    else
    { return deps_graph[from] -> lookupDependency(to); }
  }

  __host__ int getDepLength (const int from) const
  {
    if (deps_graph[from] == nullptr) 
    { return 0; }
    else
    { return deps_graph[from] -> length();}
  }

  __host__ void flattenDep (const int from, int * deps) const
  {
    if (deps_graph[from] == nullptr) 
    { deps[0] = 0; }
    else
    { deps[0] = getDepLength(from); deps_graph[from] -> flatten(&deps[1]); }
  }

  __host__ int getDepCount (const int to) const
  {
    int sum = 0;
    for (int i = 0; i < to; i++)
    {
      if (getDep(i, to) > no_dep) { sum++; }
    }
    return sum;
  }

  __host__ unsigned long long int getFops () const
  {
    return fops;
  }

  __host__ void print() const
  {
    for (int i = 0; i < length; i++)
    {
      printf("Inst %d: ", i);
      ops_list[i].print();
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