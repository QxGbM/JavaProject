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
    const int *e0_dim = e0 -> getDim3(true);
    const struct multi_level_index *e0_id = new multi_level_index (a_id -> levels, a_id -> ns, i * nx + i, 0, e0_dim);

    const struct multi_level_index **l0 = new const struct multi_level_index *[1]{ e0_id };
    struct ops_chain *p0 = new ops_chain(getrf, 1, l0);

    if (e0 -> element_type == hierarchical) 
    { p0 -> child = get_ops_h_getrf (e0 -> get_element_hierarchical(), e0_id); }

    for (int j = i + 1; j < nx; j++)
    {
      const struct h_matrix_element <T> *e1 = (a -> elements)[i * nx + j];
      const int *e1_dim = e1 -> getDim3(true);
      const struct multi_level_index *e1_id = new multi_level_index(a_id -> levels, a_id -> ns, i * nx + j, 0, e1_dim);
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
      const int *e1_dim = e1 -> getDim3(true);
      const struct multi_level_index *e1_id = new multi_level_index(a_id -> levels, a_id -> ns, j * nx + i, 0, e1_dim);
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

        const int *e1_dim = e1 -> getDim3(true);
        const int *e2_dim = e2 -> getDim3(true);
        const int *e3_dim = e3 -> getDim3(true);

        const struct multi_level_index *e1_id = new multi_level_index(a_id -> levels, a_id -> ns, j * nx + k, 0, e1_dim);
        const struct multi_level_index *e2_id = new multi_level_index(a_id -> levels, a_id -> ns, j * nx + i, 0, e2_dim);
        const struct multi_level_index *e3_id = new multi_level_index(a_id -> levels, a_id -> ns, i * nx + k, 0, e3_dim);

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
  const int nx_a = a -> nx, ny_a = a -> ny, n = (nx_a > ny_a) ? ny_a : nx_a;
  const int nx_b = (b_id -> dim)[0], ny_b = (b_id -> dim)[1], ld = (b_id -> dim)[2];
  int offset = b_id -> offset;

  for (int i = 0; i < n; i++)
  {
    const struct h_matrix_element <T> *e0 = (a -> elements)[i * nx_a + i];
    const int *e0_dim = e0 -> getDim3(true);
    const struct multi_level_index *e0_id = new multi_level_index (a_id -> levels, a_id -> ns, i * nx_a + i, 0, e0_dim);
    const int *b0_dim = new int[3]{ nx_b, e0_dim[1], ld };
    const struct multi_level_index *b0_id = new multi_level_index (b_id -> levels, b_id -> ns, -1, offset, b0_dim);

    const struct multi_level_index **l0 = new const struct multi_level_index *[1]{ b0_id };
    const struct multi_level_index **l1 = new const struct multi_level_index *[1]{ e0_id };
    struct ops_chain *p0 = new ops_chain(trsml, 1, l0, 1, l1);
    delete[] l0;
    delete[] l1;

    if (e0 -> element_type == hierarchical)
    { p0 -> child = get_ops_h_d_trsml (b, b0_id, e0 -> get_element_hierarchical(), e0_id); }

    const int next_offset = (offset += ld * e0_dim[1]);

    delete[] e0_dim;
    delete e0_id;


    for (int j = i + 1; j < ny_a; j++)
    {
      const struct h_matrix_element <T> *e1 = (a -> elements)[j * nx_a + i];
      const int *e1_dim = e1 -> getDim3(true);
      const struct multi_level_index *e1_id = new multi_level_index(a_id -> levels, a_id -> ns, j * nx_a + i, 0, e1_dim);
      const int *b1_dim = new int[3]{ b0_dim[0], e1_dim[1], ld };
      const struct multi_level_index *b1_id = new multi_level_index(b_id -> levels, b_id -> ns, -1, offset, b1_dim);

      const struct multi_level_index **l0 = new const struct multi_level_index *[1]{ b1_id };
      const struct multi_level_index **l1 = new const struct multi_level_index *[2]{ e1_id, b0_id };
      struct ops_chain *p1 = new ops_chain(gemm, 1, l0, 2, l1);
      delete[] l0;
      delete[] l1;

      if (e1 -> element_type == hierarchical)
      { p1 -> child = get_ops_d_h_d_gemm (b, b1_id, e1 -> get_element_hierarchical(), e1_id, b, b0_id); }

      offset += e1_dim[1];
      delete[] e1_dim;
      delete e1_id;
      delete[] b1_dim;
      delete b1_id;


      p0->hookup(p1);
    }

    delete[] b0_dim;
    delete b0_id;

    offset = next_offset;

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