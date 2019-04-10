
#ifndef _DEV_HIERARCHICAL_CUH
#define _DEV_HIERARCHICAL_CUH

#include <pspl.cuh>

template <class T> class dev_hierarchical 
{
private:

  int nx;
  int ny;
  dev_h_element <T> * elements;

public:
  
  __host__ dev_hierarchical (const int nx_in, const int ny_in)
  {
    nx = nx_in;
    ny = ny_in;
    elements = new dev_h_element <T> [nx * ny];
    for (int y = 0; y < ny; y++)
    {
      for (int x = 0; x < nx; x++)
      { setElement(nullptr, empty, x, y); }
    }
  }

  __host__ ~dev_hierarchical ()
  {
    delete[] elements;
  }

  __host__ void setElement (void * M, const element_t type, const int x, const int y) 
  {
    if (x < nx && y < ny)
    { elements[y * nx + x] = * new dev_h_element <T>(M, type); }
  }

  __host__ h_index * getRootIndex () const
  {
    return new h_index (0, nullptr, 0, this);
  }

  __host__ bool dimIntegrityCheck () const
  { 
    for (int y = 0; y < ny; y++)
    {
      const int rows = elements[y * nx].getNy();
      for (int x = 1; x < nx; x++)
      {
        const int rows_x = elements[y * nx + x].getNy();
        if (rows != rows_x) 
        { printf("-- Unmatched Dimensions: (%d, %d) with (%d, 0). --\n\n", y, x, y); return false; }
      }
    }

    for (int x = 0; x < nx; x++)
    {
      const int cols = elements[x].getNx();
      for (int y = 1; y < ny; y++)
      {
        const int cols_y = elements[y * nx + x].getNx();
        if (cols != cols_y)
        { printf("-- Unmatched Dimensions: (%d, %d) with (0, %d). --\n\n", y, x, x); return false; }
      }
    }

    return true;
  }

  __host__ int getNx () const
  {
    if (dimIntegrityCheck())
    {
      int n = 0;
      for (int i = 0; i < nx; i++)
      { n += elements[i].getNx(); }
      return n;
    }
    return 0;
  }

  __host__ int getNy () const
  {
    if (dimIntegrityCheck())
    {
      int n = 0;
      for (int i = 0; i < ny; i++)
      { n += elements[i * nx].getNy(); }
      return n;
    }
    return 0;
  }

  __host__ T getElement (const int x_in, const int y_in) const
  {
    int y = 0, x = 0, r = 0, c = 0;

    while (y < ny && x < nx)
    {
      int rs = elements[y * nx + x].getNy(), cs = elements[y * nx + x].getNx();
      if (r + rs <= y_in)
      { r += rs; y++; }
      else if (c + cs <= x_in)
      { c += cs; x++; }
      else
      { return elements[y * nx + x].getElement(x_in - c, y_in - r); }
    }

    return 0;
  }

  __host__ dev_dense <T> * convertToDense() const
  {
    const int nx_d = getNx(), ny_d = getNy();
    if (nx_d > 0 && ny_d > 0)
    {
      dev_dense <T> * d = new dev_dense <T> (nx_d, ny_d);
      T * elements = d -> getElements();
      for (int y = 0; y < ny_d; y++)
      {
        for (int x = 0; x < nx_d; x++)
        { elements[y * nx_d + x] = getElement(x, y); }
      }
      return d;
    }
    else
    {
      return nullptr;
    }
  }

  __host__ const h_ops_tree * generateOps_GETRF (const bool transposed = false) const
  {
    const h_index * root = getRootIndex();
    h_ops_tree * ops = new h_ops_tree(getrf, root, getNx(), getNy(), 0, transposed), * tree = generateOps_GETRF(root, transposed);
    ops -> hookup_child(tree);
    delete root;
    return ops;
  }

  __host__ h_ops_tree * generateOps_GETRF (const h_index *self, const bool self_T) const
  {
    h_ops_tree * ops = nullptr;
    for (int i = 0; i < nx && i < ny; i++)
    {
      const h_index * index_i = self -> child(i * nx + i);
      h_ops_tree * ops_i = elements[i * nx + i].generateOps_GETRF(index_i, self_T);

      for (int j = i + 1; j < nx; j++)
      {
        const h_index * index_j = self -> child(i * nx + j);
        if (self_T)
        {
          h_ops_tree * ops_j = elements[i * nx + i].generateOps_TRSMR(index_i, self_T, &elements[i * nx + j], index_j, self_T);
          ops_i -> hookup_next(ops_j);
        }
        else
        {
          h_ops_tree * ops_j = elements[i * nx + i].generateOps_TRSML(index_i, self_T, &elements[i * nx + j], index_j, self_T);
          ops_i -> hookup_next(ops_j);
        }
        delete index_j;
      }

      for (int j = i + 1; j < ny; j++)
      {
        const h_index * index_j = self -> child(j * nx + i);
        if (self_T)
        {
          h_ops_tree * ops_j = elements[i * nx + i].generateOps_TRSML(index_i, self_T, &elements[j * nx + i], index_j, self_T);
          ops_i -> hookup_next(ops_j);
        }
        else
        {
          h_ops_tree * ops_j = elements[i * nx + i].generateOps_TRSMR(index_i, self_T, &elements[j * nx + i], index_j, self_T);
          ops_i -> hookup_next(ops_j);
        }
        delete index_j;
      }

      delete index_i;

      for (int j = i + 1; j < ny; j++)
      {
        for (int k = i + 1; k < nx; k++)
        {
          const h_index * index_j = self -> child(j * nx + i), * index_k = self -> child(i * nx + k), * index_m = self -> child(j * nx + k);
          if (self_T)
          {
            h_ops_tree * ops_m = elements[j * nx + k].generateOps_GEMM(index_m, self_T, &elements[i * nx + k], index_k, self_T, &elements[j * nx + i], index_j, self_T);
            ops_i -> hookup_next(ops_m);
          }
          else
          {
            h_ops_tree * ops_m = elements[j * nx + k].generateOps_GEMM(index_m, self_T, &elements[j * nx + i], index_j, self_T, &elements[i * nx + k], index_k, self_T);
            ops_i -> hookup_next(ops_m);
          }
          delete index_j; delete index_k; delete index_m;
        }
      }

      if (ops == nullptr)
      { ops = ops_i; }
      else
      { ops -> hookup_next (ops_i); }
    }
    return ops;
  }

  __host__ h_ops_tree * generateOps_TRSML (const h_index *self, const bool self_T, const dev_hierarchical <T> *B, const h_index *index_b, const bool B_T) const
  {
    h_ops_tree * ops = nullptr;
    if ((self_T ? nx : ny) != (B_T ? B -> nx : B -> ny)) 
    { printf("Matrices are partitioned differently in H-H TRSML.\n"); return nullptr; }

    for (int i = 0; i < nx && i < ny; i++)
    {
      const h_index * index_i = self -> child(i * nx + i);
      h_ops_tree * ops_i = nullptr;

      for (int j = 0; j < (B_T ? B -> ny : B -> nx) ; j++)
      {
        const h_index * index_bj = B_T ? index_b -> child(j * (B -> nx) + i) : index_b -> child(i * (B -> nx) + j);
        const dev_h_element <T> * e_bj = B_T ? &(B -> elements)[j * (B -> nx) + i] : &(B -> elements)[i * (B -> nx) + j];
        h_ops_tree * ops_j = elements[i * nx + i].generateOps_TRSML(index_i, self_T, e_bj, index_bj, B_T);

        for (int k = i + 1; k < (self_T ? nx : ny) && ops_j != nullptr; k++)
        {
          const h_index * index_k = self_T ? self -> child(i * nx + k) : self -> child(k * nx + i);
          const dev_h_element <T> * e_k = self_T ? &elements[i * nx + k] : &elements[k * nx + i];

          const h_index * index_bk = B_T ? index_b -> child(j * (B -> nx) + k) : index_b -> child(k * (B -> nx) + j);
          const dev_h_element <T> * e_bk = B_T ? &(B -> elements)[j * (B -> nx) + k] : &(B -> elements)[k * (B -> nx) + i];

          if (B_T)
          {
            h_ops_tree * ops_k = e_bk -> generateOps_GEMM(index_bk, B_T, e_bk, index_bj, B_T, e_k, index_k, self_T);
            ops_j->hookup_next(ops_k);
          }
          else
          {
            h_ops_tree * ops_k = e_bk -> generateOps_GEMM(index_bk, B_T, e_k, index_k, self_T, e_bk, index_bj, B_T);
            ops_j->hookup_next(ops_k);
          }

          delete index_k; delete index_bk;
        }

        delete index_bj;

        if (ops_i == nullptr)
        { ops_i = ops_j; }
        else
        { ops_i -> hookup_next(ops_j); }
      }
      delete index_i;

      if (ops == nullptr)
      { ops = ops_i; }
      else
      { ops -> hookup_next(ops_i); }
    }
    return ops;
  }

  __host__ h_ops_tree * generateOps_TRSML (const h_index *self, const bool self_T, const dev_dense <T> *B, const h_index *index_b, const bool B_T) const
  {
    h_ops_tree * ops = nullptr;
    int offset = index_b -> getOffset();

    for (int i = 0; i < nx && i < ny; i++)
    {
      const h_index * index_i = self -> child(i * nx + i), * index_bi = index_b -> child(-1, offset);
      h_ops_tree * ops_i = elements[i * nx + i].generateOps_TRSML(index_i, self_T, B, index_bi, B_T);
      const int next_offset = (offset += elements[i * nx + i].getNy() * B -> getLd());
      delete index_i;

      for (int j = i + 1; j < ny; j++)
      {
        const h_index * index_j = self -> child(j * nx + i), *index_bj = index_b -> child(-1, offset);
        h_ops_tree * ops_j = elements[j * nx + i].generateOps_GEMM_A(B, index_bj, B_T, index_j, self_T, B, index_bi, B_T);
        delete index_j; delete index_bj;
        offset += elements[j * nx + i].getNy() * B -> getLd();
        ops_i -> hookup_next(ops_j);
      }

      delete index_bi;
      offset = next_offset;

      if (ops == nullptr)
      { ops = ops_i; }
      else
      { ops -> hookup_next(ops_i); }
    }
    return ops;
  }

  __host__ h_ops_tree * generateOps_TRSML_B (const h_index *self, const bool self_T, const dev_dense <T> *B, const h_index *index_b, const bool B_T) const
  {
    h_ops_tree * ops = nullptr;
    int offset = index_b -> getOffset();
    for (int i = 0; i < ny; i++)
    {
      const h_index *index_bi = index_b -> child (-1, offset);
      for (int j = 0; j < nx; j++)
      {
        const h_index *index_j = self -> child(i * nx + j);
        h_ops_tree * ops_j = elements[i * nx + j].generateOps_TRSML_B (index_j, self_T, B, index_bi, B_T);
        if (ops == nullptr)
        { ops = ops_j; }
        else
        { ops -> hookup_next(ops_j); }
      }
      delete index_bi;

      int offset_iter = offset + elements[i * nx].getNy() * (B -> getLd());
      for (int j = i + 1; j < ny; j++)
      {
        const h_index *index_bj = index_b -> child(-1, offset_iter);
        h_ops_tree * ops_j = nullptr;
        for (int k = 0; k < nx; k++)
        {
          const h_index *index_jk = self -> child(j * nx + k);
          const h_index *index_ik = self -> child(i * nx + k);
          h_ops_tree * ops_k = elements[j * nx + k].generateOps_GEMM (index_jk, self_T, B, index_bj, B_T, &elements[i * nx + k], index_ik, self_T);
          delete index_jk; delete index_ik;
          if (ops == nullptr)
          { ops_j = ops_k; }
          else
          { ops_j -> hookup_next(ops_k); }
        }
        delete index_bj;
        ops -> hookup_next(ops_j);
      }
      offset += elements[i * nx].getNy() * (B -> getLd() + 1);
    }
    
    return ops;
  }

  __host__ h_ops_tree * generateOps_TRSMR (const h_index *self, const bool self_T, const dev_hierarchical <T> *B, const h_index *index_b, const bool B_T) const
  {
    h_ops_tree * ops = nullptr;
    if (nx != B -> nx)
    { printf("Matrices are partitioned differently in H-H TRSMR.\n"); return nullptr; }

    for (int i = 0; i < nx && i < ny; i++)
    {
      const h_index * index_i = self -> child(i * nx + i);
      h_ops_tree * ops_i = nullptr;

      for (int j = 0; j < B -> ny; j++)
      {
        const h_index * index_bj = index_b -> child(j * (B -> nx) + i);
        h_ops_tree * ops_j = elements[i * nx + i].generateOps_TRSMR(index_i, self_T, &(B -> elements)[j * (B -> nx) + i], index_bj, B_T);

        for (int k = i + 1; k < nx && ops_j != nullptr; k++)
        {
          const h_index * index_k = self -> child(i * nx + k), *index_bk = index_b -> child(j * (B -> nx) + k);
          h_ops_tree * ops_k = (B -> elements[j * (B -> nx) + k]).generateOps_GEMM(index_bk, self_T, &(B -> elements)[j * (B -> nx) + i], index_bj, B_T, &elements[i * nx + k], index_k, self_T);
          delete index_k; delete index_bk;
          ops_j -> hookup_next(ops_k);
        }

        delete index_bj;

        if (ops_i == nullptr)
        { ops_i = ops_j; }
        else
        { ops_i -> hookup_next(ops_j); }
      }
      delete index_i;

      if (ops == nullptr)
      { ops = ops_i; }
      else
      { ops -> hookup_next(ops_i); }
    }
    return ops;
  }

  __host__ h_ops_tree * generateOps_TRSMR(const h_index *self, const bool self_T, const dev_dense <T> *B, const h_index *index_b, const bool B_T) const
  {
    h_ops_tree * ops = nullptr;
    int offset = index_b -> getOffset();

    for (int i = 0; i < nx && i < ny; i++)
    {
      const h_index * index_i = self -> child(i * nx + i), *index_bi = index_b -> child(-1, offset);
      h_ops_tree * ops_i = elements[i * nx + i].generateOps_TRSMR(index_i, self_T, B, index_bi, B_T);
      const int next_offset = (offset += elements[i * nx + i].getNx());
      delete index_i;

      for (int j = i + 1; j < nx; j++)
      {
        const h_index * index_j = self -> child(i * nx + j), *index_bj = index_b -> child(-1, offset);
        h_ops_tree * ops_j = elements[j * nx + i].generateOps_GEMM_B(B, index_bj, B_T, B, index_bi, B_T, index_j, self_T);
        delete index_j; delete index_bj;
        offset += elements[j * nx + i].getNy();
        ops_i -> hookup_next(ops_j);
      }

      delete index_bi;
      offset = next_offset;

      if (ops == nullptr)
      { ops = ops_i; }
      else
      { ops -> hookup_next(ops_i); }
    }
    return ops;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const bool self_T, const dev_dense <T> *A, const h_index *index_a, const bool A_T, const dev_dense <T> *B, const h_index *index_b, const bool B_T) const
  {
    h_ops_tree * ops = nullptr;
    if (A -> getNx() != B -> getNy()) 
    { printf("Unmatched Dimensions in H-GEMM.\n"); return nullptr; }

    int offset_a = index_a -> getOffset();
    for (int i = 0; i < ny; i++)
    {
      const h_index * index_ai = index_a -> child(-1, offset_a);
      int offset_b = index_b -> getOffset();

      for (int j = 0; j < nx; j++)
      {
        const h_index * index_m = self -> child(i * nx + j), * index_bj = index_b -> child(-1, offset_b);
        h_ops_tree * ops_m = elements[i * nx + j].generateOps_GEMM(index_m, self_T, A, index_ai, A_T, B, index_bj, B_T);

        delete index_m; delete index_bj;
        offset_b += elements[i * nx + j].getNx();

        if (ops == nullptr)
        { ops = ops_m; }
        else
        { ops -> hookup_next(ops_m); }
      }

      delete index_ai;
      offset_a += elements[i * nx].getNy() * A -> getLd();
    }
    return ops;
  }

  __host__ T * lookup (const h_index * index, const int level_self = 0) const
  {
    if (index == nullptr || index -> getLevels() <= level_self)
    {
      return nullptr;
    }
    else if (index -> getLevels() == level_self + 1)
    {
      const dev_dense <T> *d = elements[index -> getIndex(level_self)].getElementDense();
      return (d == nullptr) ? nullptr : d -> getElements(index -> getOffset());
    }
    else
    {
      const dev_hierarchical <T> *h = elements[index -> getIndex(level_self)].getElementHierarchical();
      return (h == nullptr) ? nullptr : h -> lookup(index, level_self + 1);
    }
  }

  __host__ void print(const h_index * index_in) const
  {
    for (int i = 0; i < ny * nx; i++)
    {
      const h_index * i_index = index_in->child(i);
      elements[i].print(i_index);
      delete i_index;
    }
  }

  __host__ void print() const
  {
    const h_index * root = getRootIndex();
    print(root);
    delete root;
  }

  __host__ void loadTestMatrix (const int levels, const int dim, const int block_size, const int x_start = 0, const int y_start = 0)
  {
    for (int y = 0, y_offset = 0; y < ny; y++)
    {
      for (int x = 0, x_offset = 0; x < nx; x++)
      {
        if (x == y && levels > 0)
        { 
          dev_hierarchical <T> *e = new dev_hierarchical <T> (dim, dim);
          e -> loadTestMatrix(levels - 1, dim, block_size, x_offset, y_offset); 
          setElement(e, hierarchical, x, y);
          x_offset += e -> getNx();
        }
        else
        {
          int l = block_size, cl = levels; 
          while (cl > 0) { l *= dim; cl--; }
          dev_dense <T> *e = new dev_dense <T> (l, l);
          e -> loadTestMatrix(x_offset, y_offset);
          setElement(e, dense, x, y);
          x_offset += e -> getNx();
        }
      }
      y_offset += elements[y * nx].getNy();
    }

  }

  __host__ void loadTestMatrix2 (const int levels, const int dim, const int block_size, const int rank)
  {
    for (int y = 0; y < ny; y++)
    {
      for (int x = 0; x < nx; x++)
      {
        if (x == y && levels > 0)
        {
          dev_hierarchical <T> *e = new dev_hierarchical <T>(dim, dim);
          e -> loadTestMatrix2(levels - 1, dim, block_size, rank);
          setElement(e, hierarchical, x, y);
        }
        else
        {
          int l = block_size, cl = levels;
          while (cl > 0) { l *= dim; cl--; }
          if (levels > 0)
          {
            dev_low_rank <T> *e = new dev_low_rank <T> (l, l, rank);
            setElement(e, low_rank, x, y);
          }
          else
          {
            dev_dense <T> *e = new dev_dense <T> (l, l);
            setElement(e, dense, x, y);
          }

        }
      }
    }

  }

};

#endif