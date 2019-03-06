#ifndef _GET_OPS_CUH
#define _GET_OPS_CUH

#include <ops.cuh>
#include <dev_hierarchical.cuh>

template <class matrixEntriesT> 
__host__ struct ops_chain * get_ops_hgetrf 
  (const struct dev_hierarchical <matrixEntriesT> *a) 
{
  struct ops_chain *ops = nullptr;
  const int nx = a -> nx, ny = a -> ny, n = (nx > ny) ? ny : nx;
  for (int i = 0; i < n; i++)
  {
    const struct h_matrix_element <matrixEntriesT> *e0 = (a -> elements)[i * nx + i], *e1, *e2;
    struct ops_chain *p0 = new ops_chain(getrf, 1, &(e0 -> index)), *p1;
    if (e0 -> element_type == hierarchical) 
    { p0 -> child = get_ops_hgetrf (e0 -> get_element_hierarchical()); }

    for (int j = i + 1; j < nx; j++)
    {
      e1 = (a -> elements)[i * nx + j];
      p1 = new ops_chain(trsml, 1, &(e1 -> index), 1, &(e0 -> index));

      if (e0 -> element_type == hierarchical) 
      { p1 -> child = get_ops_htrsml (e1, e0 -> get_element_hierarchical()); }
      p0 -> hookup(p1);
    }

    for (int j = i + 1; j < ny; j++)
    {
      e2 = (a -> elements)[j * nx + i];
      p1 = new ops_chain(trsmr, 1, &(e2 -> index), 1, &(e0 -> index));

      if (e0 -> element_type == hierarchical) 
      { p1 -> child = get_ops_htrsmr (e2, e0 -> get_element_hierarchical()); }
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

        if (e0 -> element_type == hierarchical) 
        { p1 -> child = get_ops_hgemm (e0 -> get_element_hierarchical(), e1, e2); }
        p0 -> hookup(p1);
      }
    }

    if (ops == nullptr) { ops = p0; }
    else { ops -> hookup(p0); }
  }
  return ops;
}

template <class matrixEntriesT> 
__host__ struct ops_chain * get_ops_htrsml 
  (const struct h_matrix_element <matrixEntriesT> *b, const struct dev_hierarchical <matrixEntriesT> *a)
{
  struct ops_chain *ops = nullptr;
  const int nx = a -> nx, ny = a -> ny;
  const struct dev_dense <matrixEntriesT> *d = b -> get_element_dense();
  const struct dev_low_rank <matrixEntriesT> *lr = b -> get_element_low_rank();
  const struct dev_hierarchical <matrixEntriesT> *h = b -> get_element_hierarchical();

  if (d != nullptr) 
  {
    
  }
  else if (lr != nullptr) 
  {
    // TODO
  }
  else if (h != nullptr) 
  {
    // TODO
  }

  return ops;
}

template <class matrixEntriesT>
__host__ struct ops_chain * get_ops_htrsmr 
  (const struct h_matrix_element <matrixEntriesT> *b, const struct dev_hierarchical <matrixEntriesT> *a)
{
  struct ops_chain *ops = nullptr;
  const int nx = a -> nx, ny = a -> ny;
  return ops;
}

template <class matrixEntriesT> 
__host__ struct ops_chain * get_ops_hgemm 
  (const struct dev_hierarchical <matrixEntriesT> *a, const struct h_matrix_element <matrixEntriesT> *b, const struct h_matrix_element <matrixEntriesT> *c) 
{
  struct ops_chain *ops = nullptr;
  const int nx = a -> nx, ny = a -> ny;
  return ops;
}

#endif