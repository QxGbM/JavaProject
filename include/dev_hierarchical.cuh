
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

  
  __host__ inline int getX () const
  { return nx; }

  __host__ inline int getY () const
  { return ny; }

  __host__ inline dev_h_element <T> * getBlock (const int x, const int y) const
  { return (x < nx && y < ny) ? &elements[y * nx + x] : nullptr; }

  __host__ void setElement (void * M, const element_t type, const int x, const int y) 
  {
    if (x < nx && y < ny)
    { elements[y * nx + x].setElement(M, type); }
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
    { return nullptr; }
  }

  __host__ h_ops_tree * generateOps_GETRF (const h_index * self) const
  {
    const int x = getNx(), y = getNy(), ld = 0;
    h_ops_tree * op = new h_ops_tree (getrf, self, x, y, ld);

    int n = nx > ny ? ny : nx, * child_offset = new int[n];
    child_offset[0] = 0;
    for (int i = 1; i < n; i++)
    { child_offset[i] = child_offset[i - 1] + (nx - i + 1) * (ny - i + 1); }

    op -> resizeChildren(child_offset[n - 1] + (nx - n + 1) * (ny - n + 1));

#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
      const h_index * index_i = self -> child(i * nx + i);
      op -> setChild(elements[i * nx + i].generateOps_GETRF(index_i), child_offset[i]);

#pragma omp parallel for
      for (int j = i + 1; j < nx; j++)
      {
        const h_index * index_j = self -> child(i * nx + j);
        op -> setChild(elements[i * nx + i].generateOps_TRSML(index_i, &elements[i * nx + j], index_j), child_offset[i] + j - i);
        delete index_j;
      }

#pragma omp parallel for
      for (int j = i + 1; j < ny; j++)
      {
        const h_index * index_j = self -> child(j * nx + i);
        op -> setChild(elements[i * nx + i].generateOps_TRSMR(index_i, &elements[j * nx + i], index_j), child_offset[i] + (nx - i) + j - i - 1);
        delete index_j;
      }

      delete index_i;

#pragma omp parallel for
      for (int j = i + 1; j < ny; j++)
      {
#pragma omp parallel for
        for (int k = i + 1; k < nx; k++)
        {
          const h_index * index_j = self -> child(j * nx + i), * index_k = self -> child(i * nx + k), * index_m = self -> child(j * nx + k);
          op -> setChild(elements[j * nx + k].generateOps_GEMM(index_m, &elements[j * nx + i], index_j, false, &elements[i * nx + k], index_k, false), child_offset[i] + (nx + ny - 2 * i - 1) + (j - i - 1) * (nx - i - 1) + (k - i - 1));
          delete index_j; delete index_k; delete index_m;
        }
      }
    }

    delete child_offset;
    return op;
  }

  __host__ h_ops_tree * generateOps_TRSML (const h_index *self, const dev_dense <T> *B, const h_index *index_b) const
  {
    const int x_m = getNx(), y_m = getNy(), x_b = B -> getNx(), y_b = B -> getNy(), y = y_b > y_m ? y_m : y_b, ld_m = 0, ld_b = B -> getLd();
    h_ops_tree * op = new h_ops_tree (trsml, index_b, self, x_b, y, x_m, ld_b, ld_m);

    int offset = index_b -> getOffset();
    for (int i = 0; i < nx && i < ny; i++)
    {
      const h_index * index_i = self -> child(i * nx + i), * index_bi = index_b -> child(-1, offset);
      op -> setChild(elements[i * nx + i].generateOps_TRSML(index_i, B, index_bi));
      const int next_offset = (offset += elements[i * nx + i].getNy() * B -> getLd());
      delete index_i;

      for (int j = i + 1; j < ny; j++)
      {
        const h_index * index_j = self -> child(j * nx + i), *index_bj = index_b -> child(-1, offset);
        op -> setChild(B -> generateOps_GEMM(index_bj, &elements[j * nx + i], index_j, false, B, index_bi, false));
        delete index_j; delete index_bj;
        offset += elements[j * nx + i].getNy() * B -> getLd();
      }

      delete index_bi;
      offset = next_offset;
    }
    return op;
  }

  __host__ h_ops_tree * generateOps_TRSML (const h_index *self, const dev_low_rank <T> *B, const h_index *index_b) const
  {
    const int x_m = getNx(), y_m = getNy(), x_b = B -> getNx(), y_b = B -> getNy(), y = y_b > y_m ? y_m : y_b, ld_m = 0, ld_b = B -> getLd_UxS();
    h_ops_tree * op = new h_ops_tree (trsml_lr, index_b, self, x_b, y, x_m, ld_b, ld_m, false);

    int offset = index_b -> getOffset();
    for (int i = 0; i < nx && i < ny; i++)
    {
      const h_index * index_i = self -> child(i * nx + i), * index_bi = index_b -> child(-1, offset);
      op -> setChild(elements[i * nx + i].generateOps_TRSML (index_i, B, index_bi));
      const int next_offset = (offset += elements[i * nx + i].getNy() * B -> getNx());
      delete index_i;

      for (int j = i + 1; j < ny; j++)
      {
        const h_index * index_j = self -> child(j * nx + i), *index_bj = index_b -> child(-1, offset);
        op -> setChild(B -> generateOps_GEMM (index_bj, &elements[j * nx + i], index_j, false, B, index_bi, false));
        delete index_j; delete index_bj;
        offset += elements[j * nx + i].getNy() * B -> getNx();
      }

      delete index_bi;
      offset = next_offset;
    }
    return op;
  }

  __host__ h_ops_tree * generateOps_TRSML (const h_index *self, const dev_hierarchical <T> *B, const h_index *index_b) const
  {
    if (ny != B -> ny) 
    { printf("Matrices are partitioned differently in H-H TRSML.\n"); return nullptr; }

    const int x_m = getNx(), y_m = getNy(), x_b = B -> getNx(), y_b = B -> getNy(), y = y_b > y_m ? y_m : y_b, ld_m = 0, ld_b = 0;
    h_ops_tree * op = new h_ops_tree (trsml, index_b, self, x_b, y, x_m, ld_b, ld_m);

    int offset = index_b -> getOffset();
    for (int i = 0; i < nx && i < ny; i++)
    {
      const h_index * index_i = self -> child(i * nx + i);

      for (int j = 0; j < B -> nx; j++)
      {
        const h_index * index_bj = index_b -> child(i * (B -> nx) + j);
        op -> setChild(elements[i * nx + i].generateOps_TRSML(index_i, &(B -> elements)[i * (B -> nx) + j], index_bj));

        for (int k = i + 1; k < ny; k++)
        {
          const h_index * index_k = self -> child(k * nx + i), * index_bk = index_b -> child(k * (B -> nx) + j);
          op -> setChild((B -> elements[k * (B -> nx) + j]).generateOps_GEMM(index_bk, &elements[k * nx + i], index_k, false, &(B -> elements)[i * (B -> nx) + j], index_bj, false));
          delete index_k, index_bk;
        }
        delete index_bj;
      }
      delete index_i;
    }
    return op;
  }

  __host__ h_ops_tree * generateOps_TRSML (const h_index *self, const dev_h_element <T> *B, const h_index *index_b) const
  {
    const dev_hierarchical <T> *h_b = B -> getElementHierarchical();
    const dev_low_rank <T> *lr_b = B -> getElementLowRank();
    const dev_dense <T> *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_TRSML (self, d_b, index_b); }
    if (lr_b != nullptr)
    { return generateOps_TRSML (self, lr_b, index_b); }
    if (h_b != nullptr)
    { return generateOps_TRSML (self, h_b, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_TRSMR (const h_index *self, const dev_dense <T> *B, const h_index *index_b) const
  {
    const int x_m = getNx(), y_m = getNy(), x_b = B -> getNx(), y_b = B -> getNy(), x = x_b > x_m ? x_m : x_b, ld_m = 0, ld_b = B -> getLd();
    h_ops_tree * op = new h_ops_tree (trsmr, index_b, self, x, y_b, x_m, ld_b, ld_m);

    int offset = index_b -> getOffset();
    for (int i = 0; i < nx && i < ny; i++)
    {
      const h_index * index_i = self -> child(i * nx + i), *index_bi = index_b -> child(-1, offset);
      op -> setChild(elements[i * nx + i].generateOps_TRSMR(index_i, B, index_bi));
      const int next_offset = (offset += elements[i * nx + i].getNx());
      delete index_i;

      for (int j = i + 1; j < nx; j++)
      {
        const h_index * index_j = self -> child(i * nx + j), *index_bj = index_b -> child(-1, offset);
        op -> setChild(B -> generateOps_GEMM (index_bj, B, index_bi, false, &elements[j * nx + i], index_j, false));
        delete index_j; delete index_bj;
        offset += elements[j * nx + i].getNx();
      }
      delete index_bi;
      offset = next_offset;
    }
    return op;
  }

  __host__ h_ops_tree * generateOps_TRSMR (const h_index *self, const dev_low_rank <T> *B, const h_index *index_b) const
  {
    const int x_m = getNx(), y_m = getNy(), x_b = B -> getNx(), y_b = B -> getNy(), x = x_b > x_m ? x_m : x_b, ld_m = 0, ld_b = B -> getLd_VT();
    h_ops_tree * op = new h_ops_tree (trsmr_lr, index_b, self, x, y_b, x_m, ld_b, ld_m, true);

    int offset = index_b -> getOffset();
    for (int i = 0; i < nx && i < ny; i++)
    {
      const h_index * index_i = self -> child(i * nx + i), *index_bi = index_b -> child(-1, offset);
      op -> setChild(elements[i * nx + i].generateOps_TRSMR (index_i, B, index_bi));
      const int next_offset = (offset += elements[i * nx + i].getNx());
      delete index_i;

      for (int j = i + 1; j < nx; j++)
      {
        const h_index * index_j = self -> child(i * nx + j), *index_bj = index_b -> child(-1, offset);
        op -> setChild(B -> generateOps_GEMM (index_bj, B, index_bi, false, &elements[j * nx + i], index_j, false));
        delete index_j; delete index_bj;
        offset += elements[j * nx + i].getNx();
      }

      delete index_bi;
      offset = next_offset;
    }
    return op;
  }

  __host__ h_ops_tree * generateOps_TRSMR (const h_index *self, const dev_hierarchical <T> *B, const h_index *index_b) const
  {
    if (nx != B -> nx)
    { printf("Matrices are partitioned differently in H-H TRSMR.\n"); return nullptr; }

    const int x_m = getNx(), y_m = getNy(), x_b = B -> getNx(), y_b = B -> getNy(), x = x_b > x_m ? x_m : x_b, ld_m = 0, ld_b = 0;
    h_ops_tree * op = new h_ops_tree (trsmr, index_b, self, x, y_b, x_m, ld_b, ld_m);

    for (int i = 0; i < nx && i < ny; i++)
    {
      const h_index * index_i = self -> child(i * nx + i);

      for (int j = 0; j < B -> ny; j++)
      {
        const h_index * index_bj = index_b -> child(j * (B -> nx) + i);
        op -> setChild(elements[i * nx + i].generateOps_TRSMR(index_i, &(B -> elements)[j * (B -> nx) + i], index_bj));

        for (int k = i + 1; k < nx; k++)
        {
          const h_index * index_k = self -> child(i * nx + k), *index_bk = index_b -> child(j * (B -> nx) + k);
          op -> setChild((B -> elements[j * (B -> nx) + k]).generateOps_GEMM(index_bk, &(B -> elements)[j * (B -> nx) + i], index_bj, false, &elements[i * nx + k], index_k, false));
          delete index_k; delete index_bk;
        }
        delete index_bj;
      }
      delete index_i;
    }
    return op;
  }

  __host__ h_ops_tree * generateOps_TRSMR (const h_index *self, const dev_h_element <T> *B, const h_index *index_b) const
  {
    const dev_hierarchical <T> *h_b = B -> getElementHierarchical();
    const dev_low_rank <T> *lr_b = B -> getElementLowRank();
    const dev_dense <T> *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_TRSMR (self, d_b, index_b); }
    if (lr_b != nullptr)
    { return generateOps_TRSMR (self, lr_b, index_b); }
    if (h_b != nullptr)
    { return generateOps_TRSMR (self, h_b, index_b); }

    return nullptr;  
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_dense <T> *A, const h_index *index_a, const bool A_T, const dev_dense <T> *B, const h_index *index_b, const bool B_T) const
  {
    const int x_m = getNx(), y_m = getNy(), x_a = A -> getNx(), y_a = A -> getNy(), x_b = B -> getNx(), y_b = B -> getNy();
    const int m = y_m > y_a ? y_a : y_m, n = x_m > x_b ? x_b : x_m, k = x_a > y_b ? y_b : x_a;
    const int ld_m = 0, ld_a = A -> getLd(), ld_b = B -> getLd();
    h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b, m, n, k, ld_m, ld_a, ld_b, A_T, B_T);

    int offset_a = index_a -> getOffset();
    for (int i = 0; i < ny; i++)
    {
      const h_index * index_ai = index_a -> child(-1, offset_a);
      int offset_b = index_b -> getOffset();

      for (int j = 0; j < nx; j++)
      {
        const h_index * index_m = self -> child(i * nx + j), * index_bj = index_b -> child(-1, offset_b);
        op -> setChild(elements[i * nx + j].generateOps_GEMM(index_m, A, index_ai, A_T, B, index_bj, B_T));

        delete index_m; delete index_bj;
        offset_b += elements[i * nx + j].getNx();
      }

      delete index_ai;
      offset_a += elements[i * nx].getNy() * A -> getLd();
    }
    return op;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_low_rank <T> *A, const h_index *index_a, const bool A_T, const dev_dense <T> *B, const h_index *index_b, const bool B_T) const
  {
    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_hierarchical <T> *A, const h_index *index_a, const bool A_T, const dev_dense <T> *B, const h_index *index_b, const bool B_T) const
  {
    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_h_element <T> *A, const h_index *index_a, const bool A_T, const dev_dense <T> *B, const h_index *index_b, const bool B_T) const
  {
    const dev_hierarchical <T> *h_a = A -> getElementHierarchical();
    const dev_low_rank <T> *lr_a = A -> getElementLowRank();
    const dev_dense <T> *d_a = A -> getElementDense();

    if (d_a != nullptr)
    { return generateOps_GEMM (self, d_a, index_a, A_T, B, index_b, B_T); }
    if (lr_a != nullptr)
    { return generateOps_GEMM (self, lr_a, index_a, A_T, B, index_b, B_T); }
    if (h_a != nullptr)
    { return generateOps_GEMM (self, h_a, index_a, A_T, B, index_b, B_T); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_dense <T> *A, const h_index *index_a, const bool A_T, const dev_low_rank <T> *B, const h_index *index_b, const bool B_T) const
  {
    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_low_rank <T> *A, const h_index *index_a, const bool A_T, const dev_low_rank <T> *B, const h_index *index_b, const bool B_T) const
  {
    const int x_m = getNx(), y_m = getNy(), x_a = A -> getNx(), y_a = A -> getNy(), x_b = B -> getNx(), y_b = B -> getNy();
    const int m = y_m > y_a ? y_a : y_m, n = x_m > x_b ? x_b : x_m, k = x_a > y_b ? y_b : x_a;
    const int ld_m = 0, ld_a = 0, ld_b = 0;
    h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b, m, n, k, ld_m, ld_a, ld_b, A_T, B_T);

    int offset_a = index_a -> getOffset();
    for (int i = 0; i < ny; i++)
    {
      const h_index * index_ai = index_a -> child(-1, offset_a);
      int offset_b = index_b -> getOffset();

      for (int j = 0; j < nx; j++)
      {
        const h_index * index_m = self -> child(i * nx + j), * index_bj = index_b -> child(-1, offset_b);
        op -> setChild(elements[i * nx + j].generateOps_GEMM(index_m, A, index_ai, A_T, B, index_bj, B_T));

        delete index_m; delete index_bj;
        offset_b += elements[i * nx + j].getNx();
      }

      delete index_ai;
      offset_a += elements[i * nx].getNy() * A -> getNx();
    }
    return op;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_hierarchical <T> *A, const h_index *index_a, const bool A_T, const dev_low_rank <T> *B, const h_index *index_b, const bool B_T) const
  {
    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_h_element <T> *A, const h_index *index_a, const bool A_T, const dev_low_rank <T> *B, const h_index *index_b, const bool B_T) const
  {
    const dev_hierarchical <T> *h_a = A -> getElementHierarchical();
    const dev_low_rank <T> *lr_a = A -> getElementLowRank();
    const dev_dense <T> *d_a = A -> getElementDense();

    if (d_a != nullptr)
    { return generateOps_GEMM (self, d_a, index_a, A_T, B, index_b, B_T); }
    if (lr_a != nullptr)
    { return generateOps_GEMM (self, lr_a, index_a, A_T, B, index_b, B_T); }
    if (h_a != nullptr)
    { return generateOps_GEMM (self, h_a, index_a, A_T, B, index_b, B_T); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_dense <T> *A, const h_index *index_a, const bool A_T, const dev_hierarchical <T> *B, const h_index *index_b, const bool B_T) const
  {
    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_low_rank <T> *A, const h_index *index_a, const bool A_T, const dev_hierarchical <T> *B, const h_index *index_b, const bool B_T) const
  {
    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_hierarchical <T> *A, const h_index *index_a, const bool A_T, const dev_hierarchical <T> *B, const h_index *index_b, const bool B_T) const
  {
    if (ny != A -> ny || nx != B -> nx || A -> nx != B -> ny)
    { printf("Matrices are partitioned differently in H-H.H GEMM.\n"); return nullptr; }

    const int x_m = getNx(), y_m = getNy(), x_a = A -> getNx(), y_a = A -> getNy(), x_b = B -> getNx(), y_b = B -> getNy();
    const int m = y_m > y_a ? y_a : y_m, n = x_m > x_b ? x_b : x_m, k = x_a > y_b ? y_b : x_a;
    const int ld_m = 0, ld_a = 0, ld_b = 0;
    h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b, m, n, k, ld_m, ld_a, ld_b, false, false);
    op -> resizeChildren(nx * ny * A -> nx);

#pragma omp parallel for
    for (int i = 0; i < ny; i++)
    {
#pragma omp parallel for
      for (int j = 0; j < nx; j++)
      {
        const h_index * index_m = self -> child(i * nx + j);
#pragma omp parallel for
        for (int k = 0; k < A -> nx; k++)
        {
          const int i_ak = i * (A -> nx) + k, i_bk = k * (B -> nx) + j;
          const h_index * index_ak = index_a -> child(i_ak), * index_bk = index_b -> child(i_bk);
          op -> setChild(elements[i * nx + j].generateOps_GEMM(index_m, &(A -> elements[i_ak]), index_ak, A_T, &(B -> elements[i_bk]), index_bk, B_T), (i * nx + j) * (A -> nx) + k);
          delete index_ak; delete index_bk;
        }
        delete index_m;
      }
    }
    return op;  
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_h_element <T> *A, const h_index *index_a, const bool A_T, const dev_hierarchical <T> *B, const h_index *index_b, const bool B_T) const
  {
    const dev_hierarchical <T> *h_a = A -> getElementHierarchical();
    const dev_low_rank <T> *lr_a = A -> getElementLowRank();
    const dev_dense <T> *d_a = A -> getElementDense();

    if (d_a != nullptr)
    { return generateOps_GEMM (self, d_a, index_a, A_T, B, index_b, B_T); }
    if (lr_a != nullptr)
    { return generateOps_GEMM (self, lr_a, index_a, A_T, B, index_b, B_T); }
    if (h_a != nullptr)
    { return generateOps_GEMM (self, h_a, index_a, A_T, B, index_b, B_T); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_dense <T> *A, const h_index *index_a, const bool A_T, const dev_h_element <T> *B, const h_index *index_b, const bool B_T) const
  {
    const dev_hierarchical <T> *h_b = B -> getElementHierarchical();
    const dev_low_rank <T> *lr_b = B -> getElementLowRank();
    const dev_dense <T> *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, A_T, d_b, index_b, B_T); }
    if (lr_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, A_T, lr_b, index_b, B_T); }
    if (h_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, A_T, h_b, index_b, B_T); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_low_rank <T> *A, const h_index *index_a, const bool A_T, const dev_h_element <T> *B, const h_index *index_b, const bool B_T) const
  {
    const dev_hierarchical <T> *h_b = B -> getElementHierarchical();
    const dev_low_rank <T> *lr_b = B -> getElementLowRank();
    const dev_dense <T> *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, A_T, d_b, index_b, B_T); }
    if (lr_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, A_T, lr_b, index_b, B_T); }
    if (h_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, A_T, h_b, index_b, B_T); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_hierarchical <T> *A, const h_index *index_a, const bool A_T, const dev_h_element <T> *B, const h_index *index_b, const bool B_T) const
  {
    const dev_hierarchical <T> *h_b = B -> getElementHierarchical();
    const dev_low_rank <T> *lr_b = B -> getElementLowRank();
    const dev_dense <T> *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, A_T, d_b, index_b, B_T); }
    if (lr_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, A_T, lr_b, index_b, B_T); }
    if (h_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, A_T, h_b, index_b, B_T); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_h_element <T> *A, const h_index *index_a, const bool A_T, const dev_h_element <T> *B, const h_index *index_b, const bool B_T) const
  {
    const dev_hierarchical <T> *h_b = B -> getElementHierarchical();
    const dev_low_rank <T> *lr_b = B -> getElementLowRank();
    const dev_dense <T> *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, A_T, d_b, index_b, B_T); }
    if (lr_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, A_T, lr_b, index_b, B_T); }
    if (h_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, A_T, h_b, index_b, B_T); }

    return nullptr;
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
      const dev_low_rank <T> *lr = elements[index->getIndex(level_self)].getElementLowRank();
      return (d == nullptr) ? (lr == nullptr ? nullptr : lr -> getElements(index -> getOffset())) : d -> getElements(index -> getOffset());
    }
    else
    {
      const dev_hierarchical <T> *h = elements[index -> getIndex(level_self)].getElementHierarchical();
      return (h == nullptr) ? nullptr : h -> lookup(index, level_self + 1);
    }
  }

  __host__ int * lookup_pivot (const h_index * index, const int level_self = 0) const
  {
    if (index == nullptr || index -> getLevels() <= level_self)
    {
      return nullptr;
    }
    else if (index -> getLevels() == level_self + 1)
    {
      const dev_dense <T> *d = elements[index -> getIndex(level_self)].getElementDense();
      return (d == nullptr) ? nullptr : d -> getPivot(index -> getOffset());
    }
    else
    {
      const dev_hierarchical <T> *h = elements[index -> getIndex(level_self)].getElementHierarchical();
      return (h == nullptr) ? nullptr : h -> lookup_pivot(index, level_self + 1);
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

  __host__ void loadTestMatrix (const int levels, const int dim, const int block_size, const int rank, const int admis, const int x_start = 0, const int y_start = 0)
  {
    int l = block_size, cl = levels;
    while (cl > 0) { l *= dim; cl--; }

    for (int y = 0, y_offset = y_start; y < ny; y++)
    {
      for (int x = 0, x_offset = x_start; x < nx; x++)
      {
        if (levels > 0 && (abs(x_offset - y_offset) < l + admis * block_size))
        { 
          dev_hierarchical <T> *e = new dev_hierarchical <T> (dim, dim);
          e -> loadTestMatrix(levels - 1, dim, block_size, rank, admis, x_offset, y_offset); 
          setElement(e, hierarchical, x, y);
          x_offset += e -> getNx();
        }
        else
        {
          if (abs(x_offset - y_offset) <= admis * block_size)
          {
            dev_dense <T> *e = new dev_dense <T> (l, l);
            e -> loadTestMatrix(x_offset, y_offset);
            setElement(e, dense, x, y);
            x_offset += e -> getNx();
          }
          else
          {
            dev_low_rank <T> *e = new dev_low_rank <T> (l, l);
            e -> loadTestMatrix (x_offset, y_offset);
            e -> adjustRank(rank);
            setElement(e, low_rank, x, y);
            x_offset += e -> getNx();
          }
        }
      }
      y_offset += elements[y * nx].getNy();
    }

  }

};

#endif