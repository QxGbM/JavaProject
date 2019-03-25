#ifndef _GET_OPS_CUH
#define _GET_OPS_CUH

#include <pspl.cuh>

template <class T> 
__host__ h_ops_tree * get_ops_h_getrf (const dev_hierarchical <T> *a, const h_index *a_id) 
{
  h_ops_tree *ops = nullptr;
  const int *a_dim = a -> getDim3(false);
  const int nx = a_dim[0], ny = a_dim[1], n = (nx > ny) ? ny : nx;
  delete[] a_dim;
  for (int i = 0; i < n; i++)
  {
    const dev_h_element <T> *e0 = a -> lookup(i * nx + i);
    const h_index *e0_id = a -> child_index(i * nx + i, a_id);
    h_ops_tree *p0 = new h_ops_tree(getrf);
    p0 -> addWR(e0_id);

    if (e0 -> getType() == hierarchical) 
    { p0 -> hookup_child (get_ops_h_getrf (e0 -> get_element_hierarchical(), e0_id)); }

    for (int j = i + 1; j < nx; j++)
    {
      const dev_h_element <T> *e1 = a -> lookup(i * nx + j);
      const h_index *e1_id = a -> child_index(i * nx + j, a_id);
      h_ops_tree *p1 = new h_ops_tree(trsml);
      p1 -> addWR(e1_id);
      p1 -> addR(e0_id);

      if (e0 -> getType() == hierarchical)
      { p1 -> hookup_child (get_ops_h_trsml (e1, e1_id, e0 -> get_element_hierarchical(), e0_id)); }

      delete e1_id;
      p0 -> hookup_next(p1);
    }

    for (int j = i + 1; j < ny; j++)
    {
      const dev_h_element <T> *e1 = a -> lookup(j * nx + i);
      const h_index *e1_id = a -> child_index(j * nx + i, a_id);
      h_ops_tree *p1 = new h_ops_tree(trsmr);
      p1 -> addWR(e1_id);
      p1 -> addR(e0_id);

      if (e0 -> getType() == hierarchical)
      { p1 -> hookup_child (get_ops_h_trsmr (e1, e1_id, e0 -> get_element_hierarchical(), e0_id)); }

      delete e1_id;
      p0 -> hookup_next(p1);
    }

    delete e0_id;

    for (int j = i + 1; j < ny; j++)
    {
      for (int k = i + 1; k < nx; k++)
      {
        const dev_h_element <T> *e1 = a -> lookup(j * nx + k);
        const dev_h_element <T> *e2 = a -> lookup(j * nx + i);
        const dev_h_element <T> *e3 = a -> lookup(i * nx + k);

        const h_index *e1_id = a -> child_index(j * nx + k, a_id);
        const h_index *e2_id = a -> child_index(j * nx + i, a_id);
        const h_index *e3_id = a -> child_index(i * nx + k, a_id);
        h_ops_tree *p1 = new h_ops_tree(gemm);
        p1 -> addWR(e1_id);
        p1 -> addR(e2_id);
        p1 -> addR(e3_id);

        if (e1 -> getType() == hierarchical)
        { p1 -> hookup_child (get_ops_h_gemm (e1 -> get_element_hierarchical(), e1_id, e2, e2_id, e3, e3_id)); }

        delete e1_id;
        delete e2_id;
        delete e3_id;
        p0 -> hookup_next(p1);
      }
    }

    if (ops == nullptr) { ops = p0; }
    else { ops -> hookup_next(p0); }
    
  }
  return ops;
}

template <class T> 
__host__ h_ops_tree * get_ops_h_trsml (const dev_h_element <T> *b, const h_index *b_id, const dev_hierarchical <T> *a, const h_index *a_id)
{
  const dev_dense <T> *d = b -> get_element_dense();
  const dev_low_rank <T> *lr = b -> get_element_low_rank();
  const dev_hierarchical <T> *h = b -> get_element_hierarchical();

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
    return get_ops_h_h_trsml (h, b_id, a, a_id);
  }
  return nullptr;
}

template <class T>
__host__ h_ops_tree * get_ops_h_d_trsml(const dev_dense <T> *b, const h_index *b_id, const dev_hierarchical <T> *a, const h_index *a_id)
{
  h_ops_tree *ops = nullptr;
  const int *a_dim = a -> getDim3(false);
  const int nx_a = a_dim[0], ny_a = a_dim[1], n = (nx_a > ny_a) ? ny_a : nx_a;
  const int nx_b = b_id -> getNx(), ld = b_id -> getLd();
  int offset = b_id -> getOffset();
  delete[] a_dim;

  for (int i = 0; i < n; i++)
  {
    const dev_h_element <T> *e0 = a -> lookup(i * nx_a + i);
    const int *e0_dim = e0 -> getDim3(true);
    const h_index *e0_id = a_id -> child(i * nx_a + i, 0, e0_dim);
    const int *b0_dim = new int[3]{ nx_b, e0_dim[1], ld };
    const h_index *b0_id = b_id -> child(-1, offset, b0_dim);

    h_ops_tree *p0 = new h_ops_tree(trsml);
    p0 -> addWR(b0_id);
    p0 -> addR(e0_id);

    if (e0 -> getType() == hierarchical)
    { p0 -> hookup_child (get_ops_h_d_trsml (b, b0_id, e0 -> get_element_hierarchical(), e0_id)); }

    const int next_offset = (offset += ld * e0_dim[1]);

    delete[] e0_dim;
    delete e0_id;

    for (int j = i + 1; j < ny_a; j++)
    {
      const dev_h_element <T> *e1 = a -> lookup(j * nx_a + i);
      const int *e1_dim = e1 -> getDim3(true);
      const h_index *e1_id = a_id -> child(j * nx_a + i, 0, e1_dim);
      const int *b1_dim = new int[3]{ b0_dim[0], e1_dim[1], ld };
      const h_index *b1_id = b_id -> child(-1, offset, b1_dim);

      h_ops_tree *p1 = new h_ops_tree(gemm);
      p1 -> addWR(b1_id);
      p1 -> addR(e1_id);
      p1 -> addR(b0_id);

      if (e1 -> getType() == hierarchical)
      { p1 -> hookup_child (get_ops_d_h_d_gemm (b, b1_id, e1 -> get_element_hierarchical(), e1_id, b, b0_id)); }

      offset += e1_dim[1];
      delete[] e1_dim;
      delete e1_id;
      delete[] b1_dim;
      delete b1_id;

      p0 -> hookup_next(p1);
    }

    delete[] b0_dim;
    delete b0_id;

    offset = next_offset;

    if (ops == nullptr) { ops = p0; }
    else { ops -> hookup_next(p0); }
  }
  return ops;
}

template <class T>
__host__ h_ops_tree * get_ops_h_lr_trsml (const dev_low_rank <T> *b, const h_index *b_id, const dev_hierarchical <T> *a, const h_index *a_id)
{
  //TODO
  return nullptr;
}

template <class T>
__host__ h_ops_tree * get_ops_h_h_trsml (const dev_hierarchical <T> *b, const h_index *b_id, const dev_hierarchical <T> *a, const h_index *a_id)
{
  h_ops_tree *ops = nullptr;
  return ops;
}


template <class T>
__host__ h_ops_tree * get_ops_h_trsmr (const dev_h_element <T> *b, const h_index *b_id, const dev_hierarchical <T> *a, const h_index *a_id)
{
  const dev_dense <T> *d = b -> get_element_dense();
  const dev_low_rank <T> *lr = b -> get_element_low_rank();
  const dev_hierarchical <T> *h = b -> get_element_hierarchical();

  if (d != nullptr)
  {
    return get_ops_h_d_trsmr(d, b_id, a, a_id);
  }
  else if (lr != nullptr)
  {
    return get_ops_h_lr_trsmr(lr, b_id, a, a_id);
  }
  else if (h != nullptr)
  {
    // TODO
  }
  return nullptr;
}

template <class T>
__host__ h_ops_tree * get_ops_h_d_trsmr(const dev_dense <T> *b, const h_index *b_id, const dev_hierarchical <T> *a, const h_index *a_id)
{
  h_ops_tree *ops = nullptr;
  const int *a_dim = a -> getDim3(false);
  const int nx_a = a_dim[0], ny_a = a_dim[1], n = (nx_a > ny_a) ? ny_a : nx_a;
  const int ny_b = b_id -> getNy(), ld = b_id -> getLd();
  int offset = b_id -> getOffset();
  delete[] a_dim;

  for (int i = 0; i < n; i++)
  {
    const dev_h_element <T> *e0 = a -> lookup(i * nx_a + i);
    const int *e0_dim = e0 -> getDim3(true);
    const h_index *e0_id = a_id -> child(i * nx_a + i, 0, e0_dim);
    const int *b0_dim = new int[3]{ e0_dim[0], ny_b, ld };
    const h_index *b0_id = b_id -> child(-1, offset, b0_dim);

    h_ops_tree *p0 = new h_ops_tree(trsmr);
    p0 -> addWR(b0_id);
    p0 -> addR(e0_id);

    if (e0->getType() == hierarchical)
    {
      p0->hookup_child(get_ops_h_d_trsmr(b, b0_id, e0->get_element_hierarchical(), e0_id));
    }

    const int next_offset = (offset += e0_dim[0]);

    delete[] e0_dim;
    delete e0_id;

    for (int j = i + 1; j < nx_a; j++)
    {
      const dev_h_element <T> *e1 = a -> lookup(i * nx_a + j);
      const int *e1_dim = e1 -> getDim3(true);
      const h_index *e1_id = a_id -> child(i * nx_a + j, 0, e1_dim);
      const int *b1_dim = new int[3]{ e1_dim[0], b0_dim[1], ld };
      const h_index *b1_id = b_id -> child(-1, offset, b1_dim);

      h_ops_tree *p1 = new h_ops_tree(gemm);
      p1 -> addWR(b1_id);
      p1 -> addR(b0_id);
      p1 -> addR(e1_id);

      if (e1 -> getType() == hierarchical)
      {
        p1 -> hookup_child(get_ops_d_d_h_gemm(b, b1_id, b, b0_id, e1->get_element_hierarchical(), e1_id));
      }

      offset += e1_dim[0];
      delete[] e1_dim;
      delete e1_id;
      delete[] b1_dim;
      delete b1_id;

      p0 -> hookup_next(p1);
    }

    delete[] b0_dim;
    delete b0_id;

    offset = next_offset;

    if (ops == nullptr) { ops = p0; }
    else { ops -> hookup_next(p0); }
  }
  return ops;
}

template <class T>
__host__ h_ops_tree * get_ops_h_lr_trsmr(const dev_low_rank <T> *b, const h_index *b_id, const dev_hierarchical <T> *a, const h_index *a_id)
{
  //TODO
  return nullptr;
}

template <class T> 
__host__ h_ops_tree * get_ops_h_gemm (const dev_hierarchical <T> *a, const h_index *a_id, 
  const dev_h_element <T> *b, const  h_index *b_id, 
  const dev_h_element <T> *c, const  h_index *c_id)
{
  const dev_dense <T> *d0 = b -> get_element_dense();
  const dev_low_rank <T> *lr0 = b -> get_element_low_rank();
  const dev_hierarchical <T> *h0 = b -> get_element_hierarchical();

  const dev_dense <T> *d1 = b->get_element_dense();
  const dev_low_rank <T> *lr1 = b->get_element_low_rank();
  const dev_hierarchical <T> *h1 = b->get_element_hierarchical();

  if (d0 != nullptr || d1 != nullptr)
  {
    return get_ops_h_d_d_gemm(a, a_id, d0, b_id, d1, c_id);
  }

  return nullptr;
}

template <class T>
__host__ h_ops_tree * get_ops_h_d_d_gemm(const dev_hierarchical <T> *a, const h_index *a_id,
  const dev_dense <T> *b, const h_index *b_id,
  const dev_dense <T> *c, const h_index *c_id)
{
  h_ops_tree *ops = nullptr;
  const int *a_dim = a -> getDim3(false);
  const int nx_a = a_dim[0], ny_a = a_dim[1], ld_b = b_id -> getLd(), ld_c = c_id -> getLd();
  const int ny_b = b_id -> getNy(), nx_c = c_id -> getNx(), k = (ny_b > nx_c) ? nx_c : ny_b;
  int offset_b = b_id -> getOffset();
  for (int i = 0; i < ny_a; i++)
  {
    int offset_c = c_id -> getOffset(), next_offset_b;
    for (int j = 0; j < nx_a; j++)
    {
      const dev_h_element <T> *e = a -> lookup(i * nx_a + j);
      const int *e_dim = e -> getDim3(true);
      const int m = e_dim[1], n = e_dim[0];
      const h_index *e_id = a_id -> child(i * nx_a + j, 0, e_dim);

      const int *b0_dim = new int[3]{ k, m, ld_b };
      const int *c0_dim = new int[3]{ n, k, ld_c };
      const h_index *b0_id = b_id -> child(-1, offset_b, b0_dim);
      const h_index *c0_id = c_id -> child(-1, offset_c, c0_dim);
      next_offset_b = offset_b + m * ld_b;
      offset_c += n;

      h_ops_tree *p = new h_ops_tree(gemm);
      p -> addWR(e_id);
      p -> addR(b0_id);
      p -> addR(c0_id);

      if (e -> getType() == hierarchical)
      {
        p -> hookup_child( get_ops_h_d_d_gemm (e -> get_element_hierarchical(), e_id, b, b0_id, c, c0_id));
      }

      delete[] e_dim;
      delete[] b0_dim;
      delete[] c0_dim;
      delete e_id;
      delete b0_id;
      delete c0_id;

      if (ops == nullptr) { ops = p; }
      else { ops -> hookup_next(p); }
    }
    offset_b = next_offset_b;
  }
  return ops;
}

template <class T>
__host__ h_ops_tree * get_ops_d_h_d_gemm (const dev_dense <T> *a, const h_index *a_id,
  const dev_hierarchical <T> *b, const h_index *b_id,
  const dev_dense <T> *c, const h_index *c_id)
{
  h_ops_tree *ops = nullptr;
  return ops;
}

template <class T>
__host__ h_ops_tree * get_ops_d_d_h_gemm(const dev_dense <T> *a, const h_index *a_id,
  const dev_dense <T> *b, const h_index *b_id,
  const dev_hierarchical <T> *c, const h_index *c_id)
{
  h_ops_tree *ops = nullptr;
  return ops;
}

#endif