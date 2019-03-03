#ifndef _DAG_CUH
#define _DAG_CUH

#include <ops.cuh>
#include <dev_hierarchical.cuh>

template <class matrixEntriesT> __host__ struct ops_chain * get_ops_hgetrf (const struct dev_hierarchical <matrixEntriesT> *a)
{
  struct ops_chain *ops = nullptr;
  int nx = a -> nx, ny = a -> ny, n = (nx > ny) ? ny : nx;
  for (int i = 0; i < n; i++)
  {
    struct h_matrix_element <matrixEntriesT> *e0 = (a -> elements)[i * nx + i], *e1, *e2;
    struct ops_chain *p0 = new ops_chain(getrf, 1, &(e0 -> index)), *p1;
    if (e0 -> element_type == hierarchical) 
    { p0 -> child = get_ops_hgetrf ((struct dev_hierarchical <matrixEntriesT> *) (e0 -> element)); }

    for (int j = i + 1; j < nx; j++)
    {
      e1 = (a -> elements)[i * nx + j];
      p1 = new ops_chain(trsml, 1, &(e1 -> index), 1, &(e0 -> index));
      // TODO: hgessm
      //if (e1 -> element_type == hierarchical) 
      //{ p1 -> child = get_ops_hgessm ((struct dev_hierarchical <matrixEntriesT> *) (e1 -> element)); }
      p0 -> hookup(p1);
    }

    for (int j = i + 1; j < ny; j++)
    {
      e2 = (a -> elements)[j * nx + i];
      p1 = new ops_chain(trsmr, 1, &(e2 -> index), 1, &(e0 -> index));
      // TODO: htstrf
      //if (e2 -> element_type == hierarchical) 
      //{ p1 -> child = get_ops_htstrf ((struct dev_hierarchical <matrixEntriesT> *) (e2 -> element)); }
      p0 -> hookup(p1);
    }

    for (int j = i + 1; j < ny; j++)
    {
      for (int k = i + 1; k < nx; k++)
      {
        e0 = (a -> elements)[j * nx + k];
        e1 = (a -> elements)[j * nx + i];
        e2 = (a -> elements)[i * nx + k];
        struct multi_level_index **in1 = new struct multi_level_index * [2]; 
        in1[0] = e1 -> index; in1[1] = e2 -> index; 
        p1 = new ops_chain(gemm, 1, &(e0 -> index), 2, in1);
        delete[] in1;
        // TODO: hgemm 
        //if (e2 -> element_type == hierarchical) 
        //{ p1 -> child = get_ops_hgemm ((struct dev_hierarchical <matrixEntriesT> *) (e0 -> element)); }
        p0 -> hookup(p1);
      }
    }

    if (ops == nullptr) { ops = p0; }
    else { ops -> hookup(p0); }
  }
  return ops;
}

enum dep_t 
{
  no_dep,
  flow_dep,
  anti_dep,
  flow_anti_dep,
  output_dep,
  flow_output_dep,
  anti_output_dep,
  flow_anti_output_dep,
};

__host__ dep_t add_dep (dep_t x, dep_t y) 
{
  int a = (int) x, b = (int) y, r = 0;
  for (int i = 4; i > 0; i = i / 2)
  {
    if (a >= i || b >= i) 
    { 
      r += i; 
      a = (a >= i) ? a - i : a;
      b = (b >= i) ? b - i : b; 
    }
  }
  return (dep_t) r; 
}

__host__ char * dep_str (dep_t x) 
{
  int a = (int) x, i = 3;
  char *str = new char[i + 1];

  str[i] = 'D'; i--;
  if (a == 0) { str[i] = 'N'; i--; }
  if (a >= 4) { str[i] = 'O'; a -= 4; i--; }
  if (a >= 2) { str[i] = 'A'; a -= 2; i--; }
  if (a >= 1) { str[i] = 'F'; a -= 1; i--; }

  while (i >= 0) { str[i] = ' '; i--; }

  return str;
}

struct dag {

  struct ops_chain *ops;
  int length;

  dep_t *dep;
  bool *dev_dep;

  int *dep_counts;
  int *dev_dep_counts;

  int *status;
  int *dev_status;

  __host__ dag (struct ops_chain * chain) {
    ops = chain;
    length = chain -> length();

    dep = new dep_t [length * length];
    build_dep();

    dep_counts = new int [length];
    status = new int [length];
    for (int i = 0; i < length; i++)
    {
      dep_counts[i] = 0;
      status[i] = 0;
      for (int j = 0; j < i; j++)
      { dep_counts[i] += (dep[i * length + j] > no_dep) ? 1 : 0; }
    }

    dev_dep = nullptr;
    dev_dep_counts = nullptr;
    dev_status = nullptr;
  }

  __host__ ~dag ()
  {
    delete ops;
    delete[] dep;
    delete[] dep_counts;
    delete[] status;

    if (dev_dep != nullptr) { cudaFree(dev_dep); }
    if (dev_dep_counts != nullptr) { cudaFree(dev_dep_counts); }
    if (dev_status != nullptr) { cudaFree(dev_status); }

    printf("-- DAG destroyed. --\n\n");
  }

  __host__ cudaError_t copyToDevice_Sync()
  {
    cudaError_t error = cudaSuccess;

    bool *t = new bool [length * length];
    for(int i = 0; i < length * length; i++) { t[i] = dep[i] > no_dep; }

    error = cudaMalloc ((void **) &dev_dep, length * length * sizeof(bool));
    error = cudaMemcpy (dev_dep, t, length * length * sizeof(bool), cudaMemcpyHostToDevice);

    delete[] t;

    if (error != cudaSuccess) { return error; }

    error = cudaMalloc ((void **) &dev_dep_counts, length * sizeof(int));
    error = cudaMemcpy (dev_dep_counts, dep_counts, length * sizeof(int), cudaMemcpyHostToDevice);

    if (error != cudaSuccess) { return error; }

    error = cudaMalloc ((void **) &dev_status, length * sizeof(int));
    error = cudaMemcpy (dev_status, status, length * sizeof(int), cudaMemcpyHostToDevice);

    if (error != cudaSuccess) { return error; }

    printf("-- DAG copied to CUDA device. --\n\n");

    return cudaSuccess;
  }

  __host__ void build_dep()
  {
    for (int i = 0; i < length; i++)
    {
      struct ops_chain *op_src = ops -> lookup(i);
      const int src_n_read_write = op_src -> n_read_write, src_n_read_only = op_src -> n_read_only; 

      for (int j = i + 1; j < length; j++)
      {
        struct ops_chain *op = ops -> lookup(j);
        const int inst_n_read_write = op -> n_read_write, inst_n_read_only = op -> n_read_only;

        dep[j * length + i] = no_dep;

        for (int k = 0; k < src_n_read_write; k++)
        {
          struct multi_level_index *dest = (op_src -> m_read_write)[k];

          for (int l = 0; l < inst_n_read_only; l++)
          {
            struct multi_level_index *in = (op -> m_read_only)[l];
            if (dest -> compare(in) >= 0) 
            { dep[j * length + i] = add_dep(flow_dep, dep[j * length + i]); break; }
          }

          for (int l = 0; l < inst_n_read_write; l++)
          {
            struct multi_level_index *in = (op -> m_read_write)[l];
            if (dest -> compare(in) >= 0) 
            { dep[j * length + i] = add_dep(output_dep, dep[j * length + i]); break; }
          }
        }

        for (int k = 0; k < inst_n_read_write; k++)
        {
          struct multi_level_index *dest = (op -> m_read_write)[k];

          for (int l = 0; l < src_n_read_only; l++)
          {
            struct multi_level_index *in = (op_src -> m_read_only)[l];
            if (dest -> compare(in) >= 0) 
            { dep[j * length + i] = add_dep(anti_dep, dep[j * length + i]); break; }
          }
        }

      }
    }
  }

  __host__ void print()
  {
    ops -> print();

    for (int i = 0; i < length; i++)
    {
      printf("%d:\t", i);
      for (int j = 0; j < length; j++)
      {
        if (j > i)
        {
          char *str = dep_str(dep[j * length + i]); 
          printf("%s ", str);
          delete[] str;
        }
        else
        { printf("     "); }
      }
      printf("\n");
    }

    printf("Dependency Counts:\n");
    for(int i = 0; i < length; i++)
    {
      printf("%d: %d", i, dep_counts[i]);
      if (i != length - 1) { printf(" | "); }
    }
    printf("\n\n");
  }
  
};


#endif