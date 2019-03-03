#ifndef _GET_OPS_CUH
#define _GET_OPS_CUH

#include <ops.cuh>
#include <dev_hierarchical.cuh>

template <class matrixEntriesT> 
__host__ struct ops_chain * get_ops_hgetrf (const struct dev_hierarchical <matrixEntriesT> *a)
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

#endif