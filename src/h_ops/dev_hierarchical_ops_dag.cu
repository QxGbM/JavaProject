

#include <definitions.cuh>
#include <h_ops/dependency.cuh>
#include <h_ops/dev_hierarchical_ops.cuh>
#include <h_ops/dev_hierarchical_ops_tree.cuh>
#include <h_ops/dev_hierarchical_ops_dag.cuh>


h_ops_dag::h_ops_dag (const h_ops_tree * ops, const int start_index, const int length_max)
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
      if (dep > dependency_t::no_dep)
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
      if (dep > dependency_t::no_dep)
      { list = new dependency_linked_list(j, dep, list); }
    }
    deps_graph_to[i] = list;
  }

}

h_ops_dag::~h_ops_dag ()
{
  for (int i = 0; i < length; i++)
  { delete deps_graph_from[i]; delete deps_graph_to[i]; }

  delete[] deps_graph_from;
  delete[] deps_graph_to;
  delete ops_list;
}

int h_ops_dag::getLength () const
{ return length; }

h_ops * h_ops_dag::getOp (const int index) const
{ return ops_list -> getChild(index); }

dependency_t h_ops_dag::getDep (const int from, const int to) const
{
  if (deps_graph_from[from] == nullptr) 
  { return dependency_t::no_dep; }
  else
  { return deps_graph_from[from] -> lookupDependency(to); }
}

dependency_linked_list * h_ops_dag::getDepList_From (const int from) const
{
  if (from >= 0 && from < length)
  { return deps_graph_from[from]; }
  else
  { return nullptr; }
}

dependency_linked_list * h_ops_dag::getDepList_To (const int to) const
{
  if (to >= 0 && to < length)
  { return deps_graph_to[to]; }
  else
  { return nullptr; }
}

int h_ops_dag::getDepCount_From (const int from) const
{
  if (from >= 0 && from < length)
  { return deps_graph_from[from] -> length(); }
  else
  { return 0; }
}

int * h_ops_dag::getDepCountList_To () const
{
  int * deps = new int [length];

#pragma omp parallel for
  for (int i = 0; i < length; i++)
  { deps[i] = deps_graph_to[i] -> length(); }

  return deps;
}

long long int h_ops_dag::getFlops (const int index) const
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

long long int h_ops_dag::getFlops_Trim (const int index) const
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

void h_ops_dag::print() const
{
  for (int i = 0; i < length; i++)
  {
    printf("Inst %d: ", i);
    h_ops_tree * to = ops_list -> getChild(i);
    to -> h_ops::print();
    for (int j = 0; j < i; j++)
    {
      dependency_t dep = getDep(j, i);
      if (dep > dependency_t::no_dep)
      { 
        printf("[%d: ", j);
        switch (dep)
        {
        case dependency_t::no_dep: break;
        case dependency_t::flow_dep: printf("FD"); break;
        case dependency_t::anti_dep: printf("AD"); break;
        case dependency_t::flow_anti_dep: printf("FAD"); break;
        case dependency_t::output_dep: printf("OD"); break;
        case dependency_t::flow_output_dep: printf("FOD"); break;
        case dependency_t::anti_output_dep: printf("AOD"); break;
        case dependency_t::flow_anti_output_dep: printf("FAOD"); break;
        }
        printf("] ");
      }
    }
    printf("Flops: %lld \n", flops[i]);
  }
  printf("Total Flops: %lld.\n\n", getFlops());
}

