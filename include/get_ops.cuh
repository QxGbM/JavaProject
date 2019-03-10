#ifndef _GET_OPS_CUH
#define _GET_OPS_CUH

#include <ops.cuh>
#include <dev_hierarchical.cuh>

template <class T> 
__host__ struct ops_chain * get_ops_h_getrf (const struct dev_hierarchical <T> *a, const struct multi_level_index *a_id) 
{
  struct ops_chain *ops = nullptr;
  const int nx = a -> nx, ny = a -> ny, n = (nx > ny) ? ny : nx;
  for (int i = 0; i < n; i++)
  {
    const struct h_matrix_element <T> *e0 = (a -> elements)[i * nx + i];
    const struct multi_level_index *e0_id = new multi_level_index (a_id -> levels, a_id -> ns, i * nx + i, 0);

    const struct multi_level_index **l0 = new const struct multi_level_index *[1]{ e0_id };
    struct ops_chain *p0 = new ops_chain(getrf, 1, l0);

    if (e0 -> element_type == hierarchical) 
    { p0 -> child = get_ops_h_getrf (e0 -> get_element_hierarchical(), e0_id); }

    for (int j = i + 1; j < nx; j++)
    {
      const struct h_matrix_element <T> *e1 = (a -> elements)[i * nx + j];
      const struct multi_level_index *e1_id = new multi_level_index(a_id -> levels, a_id -> ns, i * nx + j, 0);
      const struct multi_level_index **l1 = new const struct multi_level_index *[1]{ e1_id };

      struct ops_chain *p1 = new ops_chain(trsml, 1, l1, 1, l0);

      if (e0 -> element_type == hierarchical) 
      { p1 -> child = get_ops_h_trsml (e1, e1_id, e0 -> get_element_hierarchical(), e0_id); }

      delete e1_id;
      delete[] l1;
      p0 -> hookup(p1);
    }

    for (int j = i + 1; j < ny; j++)
    {
      const struct h_matrix_element <T> *e1 = (a -> elements)[j * nx + i];
      const struct multi_level_index *e1_id = new multi_level_index(a_id -> levels, a_id -> ns, j * nx + i, 0);
      const struct multi_level_index **l1 = new const struct multi_level_index *[1]{ e1_id };

      struct ops_chain *p1 = new ops_chain(trsmr, 1, l1, 1, l0);

      if (e0 -> element_type == hierarchical) 
      { p1 -> child = get_ops_h_trsmr (e1, e1_id, e0 -> get_element_hierarchical(), e0_id); }

      delete e1_id;
      delete[] l1;
      p0 -> hookup(p1);
    }

    delete e0_id;
    delete[] l0;

    for (int j = i + 1; j < ny; j++)
    {
      for (int k = i + 1; k < nx; k++)
      {
        const struct h_matrix_element <T> *e1 = (a -> elements)[j * nx + k];
        const struct h_matrix_element <T> *e2 = (a -> elements)[j * nx + i];
        const struct h_matrix_element <T> *e3 = (a -> elements)[i * nx + k];

        const struct multi_level_index *e1_id = new multi_level_index(a_id->levels, a_id->ns, j * nx + k, 0);
        const struct multi_level_index *e2_id = new multi_level_index(a_id->levels, a_id->ns, j * nx + i, 0);
        const struct multi_level_index *e3_id = new multi_level_index(a_id->levels, a_id->ns, i * nx + k, 0);

        const struct multi_level_index **l1 = new const struct multi_level_index *[2]{ e2_id, e3_id };
        const struct multi_level_index **l2 = new const struct multi_level_index *[1]{ e1_id };

        struct ops_chain *p1 = new ops_chain(gemm, 1, l2, 2, l1);

        if (e1 -> element_type == hierarchical) 
        { p1 -> child = get_ops_h_gemm (e1 -> get_element_hierarchical(), e1_id, e2, e2_id, e3, e3_id); }

        delete e1_id;
        delete e2_id;
        delete e3_id;
        delete[] l1;
        delete[] l2;
        p0 -> hookup(p1);
      }
    }

    if (ops == nullptr) { ops = p0; }
    else { ops -> hookup(p0); }
    
  }
  return ops;
}

template <class T> 
__host__ struct ops_chain * get_ops_h_trsml (const struct h_matrix_element <T> *b, const struct multi_level_index *b_id, 
  const struct dev_hierarchical <T> *a, const struct multi_level_index *a_id)
{
  const struct dev_dense <T> *d = b -> get_element_dense();
  const struct dev_low_rank <T> *lr = b -> get_element_low_rank();
  const struct dev_hierarchical <T> *h = b -> get_element_hierarchical();

  if (d != nullptr) 
  {
    return get_ops_h_d_trsml (d, b_id, a, a_id);
  }
  else if (lr != nullptr) 
  {
    return get_ops_h_lr_trsml (lr, b_id, a, a_id);
  }
  else if (h != nullptr) 
  {
    // TODO
  }
  return nullptr;
}

template <class T>
__host__ struct ops_chain * get_ops_h_d_trsml(const struct dev_dense <T> *b, const struct multi_level_index *b_id,
  const struct dev_hierarchical <T> *a, const struct multi_level_index *a_id)
{
  struct ops_chain *ops = nullptr;
  const int nx = a -> nx, ny = a -> ny, n = (nx > ny) ? ny : nx;
  int offset = b_id -> offset;
  const int ld = b -> ld;

  for (int i = 0; i < n; i++)
  {
    const struct h_matrix_element <T> *e0 = (a -> elements)[i * nx + i];
    const struct multi_level_index *e0_id = new multi_level_index (a_id -> levels, a_id -> ns, i * nx + i, 0);
    const struct multi_level_index *b0_id = new multi_level_index (b_id -> levels, b_id -> ns, -1, offset);

    const struct multi_level_index **l0 = new const struct multi_level_index *[1]{ b0_id };
    const struct multi_level_index **l1 = new const struct multi_level_index *[1]{ e0_id };
    struct ops_chain *p0 = new ops_chain(trsml, 1, l0, 1, l1);
    delete[] l0;
    delete[] l1;

    if (e0 -> element_type == hierarchical)
    { p0 -> child = get_ops_h_d_trsml (b, b0_id, e0 -> get_element_hierarchical(), e0_id); }

    int *dim = e0 -> getDim();
    const int next_offset = offset + ld * dim[1];

    for (int j = i + 1; j < ny; j++)
    {
      const struct h_matrix_element <T> *e1 = (a -> elements)[j * nx + i];
      const struct multi_level_index *e1_id = new multi_level_index(a_id -> levels, a_id -> ns, j * nx + i, 0);
      const struct multi_level_index *b1_id = new multi_level_index(b_id -> levels, b_id -> ns, -1, offset + ld * dim[1]);

      const struct multi_level_index **l0 = new const struct multi_level_index *[1]{ b1_id };
      const struct multi_level_index **l1 = new const struct multi_level_index *[2]{ e1_id, b0_id };
      struct ops_chain *p1 = new ops_chain(gemm, 1, l0, 2, l1);

      if (e1 -> element_type == hierarchical)
      { p1 -> child = get_ops_d_h_d_gemm (b, b1_id, e1 -> get_element_hierarchical(), e1_id, b, b0_id); }

      const int *dim_e = e1 -> getDim();
      dim[0] += dim_e[0];
      dim[1] += dim_e[1];
      delete[] dim_e;

      p0->hookup(p1);
    }

    offset = next_offset;

    delete e0_id;
    delete b0_id;
    delete[] dim;

    if (ops == nullptr) { ops = p0; }
    else { ops -> hookup(p0); }
  }
  return ops;
}

template <class T>
__host__ struct ops_chain * get_ops_h_lr_trsml(const struct dev_low_rank <T> *b, const struct multi_level_index *b_id,
  const struct dev_hierarchical <T> *a, const struct multi_level_index *a_id)
{
  //TODO
  return nullptr;
}

template <class T>
__host__ struct ops_chain * get_ops_h_h_trsml(const struct dev_hierarchical <T> *b, const struct multi_level_index *b_id,
  const struct dev_hierarchical <T> *a, const struct multi_level_index *a_id)
{
  if (a -> nx != b -> nx || a -> ny != b -> ny)
  { return nullptr; }
  else
  {
    struct ops_chain *ops = nullptr;
    const int nx = a -> nx, ny = a -> ny, n = (nx > ny) ? ny : nx;
    for (int i = 0; i < n; i++)
    {
    }
    return ops;
  }
}


template <class T>
__host__ struct ops_chain * get_ops_h_trsmr (const struct h_matrix_element <T> *b, const struct multi_level_index *b_id,
  const struct dev_hierarchical <T> *a, const struct multi_level_index *a_id)
{
  struct ops_chain *ops = nullptr;
  return ops;
}

template <class T> 
__host__ struct ops_chain * get_ops_h_gemm (const struct dev_hierarchical <T> *a, const struct multi_level_index *a_id, 
  const struct h_matrix_element <T> *b, const struct multi_level_index *b_id, 
  const struct h_matrix_element <T> *c, const struct multi_level_index *c_id)
{
  struct ops_chain *ops = nullptr;
  return ops;
}

template <class T>
__host__ struct ops_chain * get_ops_d_h_d_gemm (const struct dev_dense <T> *a, const struct multi_level_index *a_id,
  const struct dev_hierarchical <T> *b, const struct multi_level_index *b_id,
  const struct dev_dense <T> *c, const struct multi_level_index *c_id)
{
  struct ops_chain *ops = nullptr;
  return ops;
}

#endif