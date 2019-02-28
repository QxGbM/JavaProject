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
        struct multi_level_index **in1 = (struct multi_level_index **) malloc(2 * sizeof(struct multi_level_index *)); 
        in1[0] = e1 -> index; in1[1] = e2 -> index; 
        p1 = new ops_chain(gemm, 1, &(e0 -> index), 2, in1);
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

enum dep_t {
  no_dep,
  flow_dep,
  anti_dep,
  flow_anti_dep,
  output_dep,
  flow_output_dep,
  anti_output_dep,
  flow_anti_output_dep,
};

__host__ dep_t add_dep (dep_t x, dep_t y) {
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

__host__ char * dep_str (dep_t x) {
  int a = (int) x;
  char *str = (char *) malloc (4 * sizeof(char));
  if (a == 0) { sprintf(str, "  ND"); return str; }

  if (a >= 4) { str[2] = 'O'; a -= 4; } else { str[2] = ' '; }
  if (a >= 2) { str[1] = 'A'; a -= 2; } else { str[1] = ' '; }
  if (a >= 1) { str[0] = 'F'; a -= 1; } else { str[0] = ' '; }
  str[3] = 'D';

  if (str[2] == ' ') 
  { 
    if (str[1] == 'A') { str[2] = 'A'; str[1] = str[0]; str[0] = ' '; }
    else { str[2] = 'F'; str[0] = ' '; }
  }
  else if (str[1] == ' ' && str[0] == 'F')
  { str[1] = 'F'; str[0] = ' '; }

  return str;
}

struct dag {

  struct ops_chain *ops;

  int length;
  dep_t *dep;
  int *dev_dep;

  int *progress;
  int *dev_progress;

  int *status;
  int *dev_status;

  __host__ dag (struct ops_chain * chain) {
    ops = chain;
    length = chain -> length();

    dep = (dep_t *) malloc (length * length * sizeof(dep_t));
    memset ((void *) dep, 0, length * length * sizeof(dep_t));

    build_dep();

    progress = (int *) malloc (length * sizeof(int));
    memset ((void *) progress, 0, length * sizeof(int));

    for (int i = 0; i < length; i++)
    {
      for (int j = i + 1; j < length; j++)
      {
        if (dep[j * length + i] != no_dep) 
        { progress[j]++; }
      }
    }

    status = (int *) malloc (length * sizeof(int));
    memset ((void *) status, 0, length * sizeof(int));

    dev_dep = nullptr;
    dev_progress = nullptr;
    dev_status = nullptr;
  }

  __host__ ~dag ()
  {
    ops -> ~ops_chain();
    free(ops);
    free(dep);
    free(progress);
    free(status);

    if (dev_dep != nullptr) { cudaFree(dev_dep); }
    if (dev_progress != nullptr) { cudaFree(dev_progress); }
    if (dev_status != nullptr) { cudaFree(dev_status); }

    printf("-- DAG destroyed. --\n\n");
  }

  __host__ cudaError_t copyToDevice_Sync()
  {
    cudaError_t error = cudaSuccess;

    int *t = (int *) malloc (length * length * sizeof(int));
    for(int i = 0; i < length * length; i++) { t[i] = (dep[i] == no_dep) ? 0 : 1; }

    error = cudaMalloc ((void **) &dev_dep, length * length * sizeof(int));
    error = cudaMemcpy (dev_dep, t, length * length * sizeof(int), cudaMemcpyHostToDevice);

    free(t);

    if (error != cudaSuccess) { return error; }

    error = cudaMalloc ((void **) &dev_progress, length * sizeof(int));
    error = cudaMemcpy (dev_progress, progress, length * sizeof(int), cudaMemcpyHostToDevice);

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
          free(str);
        }
        else
        { printf("     "); }
      }
      printf("\n");
    }
    printf("Dependency Counts:\n");

    for(int i = 0; i < length; i++)
    {
      printf("%d: %d", i, progress[i]);
      if (i != length - 1) { printf(" | "); }
    }
    printf("\n\n");
  }
  
};


#endif