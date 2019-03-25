#ifndef _DAG_CUH
#define _DAG_CUH

#include <pspl.cuh>

class dag 
{
private:
  int length;
  dependency_t *dep;

public:

  __host__ dag (const h_ops_tree * ops) 
  {
    length = ops -> length();

    dep = new dependency_t [length * length];
    for (int i = 0; i < length; i++)
    {
      const h_ops_tree *op_src = ops -> lookup(i);
      const int src_n_read_write = op_src -> getN_WR(), src_n_read_only = op_src -> getN_R();

      for (int j = i + 1; j < length; j++)
      {
        const h_ops_tree *op = ops -> lookup(j);
        const int inst_n_read_write = op ->getN_WR(), inst_n_read_only = op -> getN_R();

        dep[j * length + i] = no_dep;

        for (int k = 0; k < src_n_read_write; k++)
        {
          const h_index *dest = op_src -> getI_WR(k);

          for (int l = 0; l < inst_n_read_only; l++)
          {
            const h_index *in = op -> getI_R(l);
            switch (dest -> compare(in)) 
            {
            case no_relation: case diff_offset_no_overlap: break;
            case diff_offset_overlapped: case same_index: case contains: case contained:
              dep[j * length + i] = (dependency_t) ((int) flow_dep | (int) dep[j * length + i]); l = inst_n_read_only;
            }
          }

          for (int l = 0; l < inst_n_read_write; l++)
          {
            const h_index *in = op -> getI_WR(l);
            switch (dest -> compare(in))
            {
            case no_relation: case diff_offset_no_overlap: break;
            case diff_offset_overlapped: case same_index: case contains: case contained:
              dep[j * length + i] = (dependency_t) ((int) output_dep | (int) dep[j * length + i]); l = inst_n_read_write;
            }
          }
        }

        for (int k = 0; k < inst_n_read_write; k++)
        {
          const h_index *dest = op -> getI_WR(k);

          for (int l = 0; l < src_n_read_only; l++)
          {
            const h_index *in = op_src -> getI_R(l);
            switch (dest -> compare(in))
            {
            case no_relation: case diff_offset_no_overlap: break;
            case diff_offset_overlapped: case same_index: case contains: case contained:
              dep[j * length + i] = (dependency_t) ((int) anti_dep | (int) dep[j * length + i]); l = src_n_read_only;
            }
          }
        }

      }
    }

  }

  __host__ ~dag ()
  {
    delete[] dep;

    printf("-- DAG destroyed. --\n\n");
  }

  __host__ void print() const
  {
    printf("Dependencies:\n");
    for (int i = 0; i < length; i++)
    {
      printf("Inst %d: \n", i);
      for (int j = 0; j < i; j++)
      {
        if (dep[i * length + j] > no_dep)
        { 
          printf("From %d: ", j);
          switch (dep[i * length + j])
          {
          case no_dep: break;
          case flow_dep: printf("Flow dependency"); break;
          case anti_dep: printf("Anti dependency"); break;
          case flow_anti_dep: printf("Flow & Anti dependency"); break;
          case output_dep: printf("Output dependency"); break;
          case flow_output_dep: printf("Flow & Output dependency"); break;
          case anti_output_dep: printf("Anti & Output dependency"); break;
          case flow_anti_output_dep: printf("Flow & Anti & Output dependency"); break;
          }
          printf("\n");
        }
      }
      printf("\n");
    }
    printf("\n");
  }
  
};


#endif