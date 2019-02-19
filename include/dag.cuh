#ifndef _DAG_CUH
#define _DAG_CUH

#include <dev_hierarchical.cuh>

enum matrix_op_t {
  nop,
  getrf,
  getrf_pivot,
  apply_pivot,
  trsm,
  gemm,
};

__host__ int calc_load (int op)
{
  int load_table[] = {1, 1, 1, 1, 1, 1};
  return load_table[op];
}

struct ops_chain {

  matrix_op_t op_type;
  struct multi_level_index *dest;
  struct multi_level_index *m1;
  struct multi_level_index *m2;

  int load;
  struct ops_chain *next;
  struct ops_chain *child;

  __host__ ops_chain (matrix_op_t opin = nop, struct multi_level_index *in0 = nullptr, struct multi_level_index *in1 = nullptr, struct multi_level_index *in2 = nullptr)
  {
    op_type = opin;
    dest = (in0 == nullptr) ? nullptr : (in0 -> clone());
    m1 = (in1 == nullptr) ? nullptr : (in1 -> clone());
    m2 = (in2 == nullptr) ? nullptr : (in2 -> clone());

    load = calc_load ((int) opin);
    next = nullptr;
    child = nullptr;
  }

  __host__ ~ops_chain ()
  {
    if (dest != nullptr)
    { dest -> ~multi_level_index(); free(dest); }
    if (m1 != nullptr)
    { m1 -> ~multi_level_index(); free(m1); }
    if (m2 != nullptr)
    { m2 -> ~multi_level_index(); free(m2); }
    if (next != nullptr)
    { next -> ~ops_chain(); free(next); }
    if (child != nullptr)
    { child -> ~ops_chain(); free(child); }
  }

  __host__ void hookup (struct ops_chain *chain)
  {
    if (next != nullptr)
    { next -> hookup(chain); }
    else
    { next = chain; }
  }

  __host__ int length ()
  {
    int l_child = (child == nullptr) ? 0 : child -> length();
    int l_next = (next == nullptr) ? 0 : next -> length();
    return 1 + l_child + l_next;
  }

  __host__ void print (const bool indent = true)
  {
    for (int i = 0; i < (dest -> levels) && indent; i++) { printf("  "); }

    switch(op_type)
    {
      case nop: printf("NOP "); break;
      case getrf: printf("GETRF "); break;
      case getrf_pivot: printf("GETRFP "); break;
      case apply_pivot: printf("PIVOT "); break;
      case trsm: printf("TRSM "); break;
      case gemm: printf("GEMM "); break;
    }

    if (dest != nullptr) { dest -> print_short(); printf(", "); }
    else { printf("_, "); }
    if (m1 != nullptr) { m1 -> print_short(); printf(", "); }
    else { printf("_, "); }
    if (m2 != nullptr) { m2 -> print_short(); }
    else { printf("_"); }
    printf("\n");

    if (child != nullptr) { child -> print(); }
    if (next != nullptr) { next -> print(); }

    if (next == nullptr && dest -> levels == 1) { printf("\n"); }
  }

};

template <class matrixEntriesT> __host__ struct ops_chain * get_ops_hgetrf (const struct dev_hierarchical <matrixEntriesT> *a)
{
  struct ops_chain *ops = nullptr;
  int nx = a -> nx, ny = a -> ny, n = (nx > ny) ? ny : nx;
  for (int i = 0; i < n; i++)
  {
    struct h_matrix_element <matrixEntriesT> *e0 = (a -> elements)[i * nx + i], *e1, *e2;
    struct ops_chain *p0 = new ops_chain(getrf, e0 -> index), *p1;
    if (e0 -> element_type == hierarchical) 
    { p0 -> child = get_ops_hgetrf ((struct dev_hierarchical <matrixEntriesT> *) (e0 -> element)); }

    for (int j = i + 1; j < nx; j++)
    {
      e1 = (a -> elements)[i * nx + j];
      p1 = new ops_chain(trsm, e1 -> index, e0 -> index);
      // TODO: htrsm 
      //if (e1 -> element_type == hierarchical) 
      //{ p1 -> child = get_ops_htrsm ((struct dev_hierarchical <matrixEntriesT> *) (e1 -> element)); }
      p0 -> hookup(p1);
    }

    for (int j = i + 1; j < ny; j++)
    {
      e2 = (a -> elements)[j * nx + i];
      p1 = new ops_chain(trsm, e2 -> index, e0 -> index);
      // TODO: htrsm 
      //if (e2 -> element_type == hierarchical) 
      //{ p1 -> child = get_ops_htrsm ((struct dev_hierarchical <matrixEntriesT> *) (e2 -> element)); }
      p0 -> hookup(p1);
    }

    for (int j = i + 1; j < ny; j++)
    {
      for (int k = i + 1; k < nx; k++)
      {
        e0 = (a -> elements)[j * nx + k];
        e1 = (a -> elements)[j * nx + i];
        e2 = (a -> elements)[i * nx + k];
        p1 = new ops_chain(gemm, e0 -> index, e1 -> index, e2 -> index);
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

struct dag {

  struct ops_chain *ops;

  __host__ dag () {
    ops = nullptr;
  }


  
};


#endif