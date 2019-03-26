#ifndef _DEV_HIERARCHICAL_OPS_DAG_CUH
#define _DEV_HIERARCHICAL_OPS_DAG_CUH

#include <pspl.cuh>

class h_ops_dag 
{
private:

  int length;
  h_ops * ops_list;
  dependency_t * deps_graph;

public:

  __host__ h_ops_dag (const h_ops_tree * ops) 
  {
    length = ops -> length();

    ops_list = new h_ops[length];
    deps_graph = new dependency_t [length * length];

    ops -> flatten(ops_list);
    memset(deps_graph, 0, length * length * sizeof(dependency_t));

    for (int i = 0; i < length; i++)
    {
      for (int j = 0; j < i; j++)
      {
        deps_graph[i * length + j] = ops_list[i].checkDependencyFrom(&ops_list[j]);
      }
    }

  }

  __host__ ~h_ops_dag ()
  {
    delete[] ops_list;
    delete[] deps_graph;

    printf("-- DAG destroyed. --\n\n");
  }

  __host__ void print() const
  {
    for (int i = 0; i < length; i++)
    {
      printf("Inst %d: ", i);
      ops_list[i].print();
      for (int j = 0; j < i; j++)
      {
        if (deps_graph[i * length + j] > no_dep)
        { 
          printf("[%d: ", j);
          switch (deps_graph[i * length + j])
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
      printf("\n\n");
    }
  }
  
};


#endif