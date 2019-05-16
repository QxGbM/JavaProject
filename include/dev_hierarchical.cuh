
#ifndef _DEV_HIERARCHICAL_CUH
#define _DEV_HIERARCHICAL_CUH

#include <pspl.cuh>

template <class T> class dev_hierarchical 
{
private:

  int nx;
  int * x_offsets;

  int ny;
  int * y_offsets;

  dev_h_element <T> * elements;

public:
  
  __host__ dev_hierarchical (const int nx_in, const int ny_in, element_t type = empty, void ** elements_in = nullptr)
  {
    nx = nx_in > 0 ? nx_in : 1;
    x_offsets = new int [nx + 1];

    ny = ny_in > 0 ? ny_in : 1;
    y_offsets = new int [ny + 1];

    elements = new dev_h_element <T> [nx * ny];
    for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++)
    { setElement((type == empty && elements_in == nullptr) ? nullptr : elements_in[y * nx + x], type, x, y); }

    updateOffsets();
  }

  __host__ ~dev_hierarchical ()
  {
    delete[] x_offsets;
    delete[] y_offsets;
    delete[] elements;
  }

  __host__ inline int getNx_blocks () const
  { return nx; }

  __host__ inline int getNy_blocks () const
  { return ny; }

  __host__ inline int getNx_abs () const
  { return x_offsets[nx]; }

  __host__ inline int getNy_abs () const
  { return y_offsets[ny]; }

  __host__ dev_h_element <T> * getBlock (const int x, const int y) const
  { return (x < nx && y < ny) ? &elements[y * nx + x] : nullptr; }

  __host__ bool updateOffsets ()
  {
    x_offsets[0] = 0; y_offsets[0] = 0;
    for (int y = 0; y < ny; y++)
    { y_offsets[y + 1] = elements[y * nx].getNy() + y_offsets[y]; }
    for (int x = 0; x < nx; x++)
    { x_offsets[x + 1] = elements[x].getNx() + x_offsets[x]; }

    for (int y = 1; y < ny; y++) for (int x = 1; x < nx; x++)
    {
      const int nx_i = elements[y * nx + x].getNx(), ny_i = elements[y * nx + x].getNy();
      if ((nx_i != x_offsets[x + 1] - x_offsets[x]) && (ny_i != y_offsets[y + 1] - y_offsets[y]))
      { return false; }
    }
    return true;
  }

  __host__ void setElement (void * M, const element_t type, const int x, const int y) 
  {
    if (x < nx && y < ny)
    { elements[y * nx + x].setElement(M, type); }
  }

  __host__ T getElement (const int y_in, const int x_in) const
  {
    int y = 0, x = 0;

    while (y < ny && y_in >= y_offsets[y + 1]) { y++; }
    while (x < nx && x_in >= x_offsets[x + 1]) { x++; }

    if (y < ny && x < nx)
    { return elements[y * nx + x].getElement(y_in - y_offsets[y], x_in - x_offsets[x]); }
    else
    { return 0; }
  }

  __host__ dev_dense <T> * convertToDense() const
  {
    const int nx_d = getNx_abs(), ny_d = getNy_abs();
    if (nx_d > 0 && ny_d > 0)
    {
      dev_dense <T> * d = new dev_dense <T> (nx_d, ny_d);
      T * d_elements = d -> getElements();
      for (int y = 0; y < ny_d; y++) for (int x = 0; x < nx_d; x++)
      { d_elements[y * nx_d + x] = getElement(y, x); }
      return d;
    }
    else
    { return nullptr; }
  }

  __host__ h_index * getRootIndex () const
  { return new h_index (this); }

  __host__ h_ops_tree * generateOps_GETRF (const h_index * self) const
  {
    h_ops_tree * op = new h_ops_tree (getrf_d, self);

    int n = nx > ny ? ny : nx, * child_offset = new int[n];
    child_offset[0] = 0;

    for (int i = 1; i < n; i++)
    { child_offset[i] = child_offset[i - 1] + (nx - i + 1) * (ny - i + 1); }

    op -> resizeChildren(child_offset[n - 1] + (nx - n + 1) * (ny - n + 1));

#pragma omp parallel for num_threads(2)
    for (int i = 0; i < n; i++)
    {
      const h_index index_i = h_index (this, self, i, i);
      h_ops_tree * op_i = elements[i * nx + i].generateOps_GETRF(&index_i);
      op -> setChild(op_i, child_offset[i]);
      delete op_i;
      const int rows = ny - i - 1, cols = nx - i - 1;

      for (int j = i + 1; j < nx; j++)
      {
        const h_index index_j = h_index (this, self, i, j);
        h_ops_tree * op_j = elements[i * nx + i].generateOps_TRSML(&index_i, &elements[i * nx + j], &index_j);
        op -> setChild(op_j, child_offset[i] + j - i);
        delete op_j;
      }

      for (int j = i + 1; j < ny; j++)
      {
        const h_index index_j = h_index (this, self, j, i);
        h_ops_tree * op_j = elements[i * nx + i].generateOps_TRSMR(&index_i, &elements[j * nx + i], &index_j);
        op -> setChild(op_j, child_offset[i] + cols + j - i);
        delete op_j;
      }

      for (int j = 0; j < rows * cols; j++)
      {
        const int row = j / cols + i + 1, col = j - (row - i - 1) * cols + i + 1;
        const h_index index_j = h_index (this, self, row, i), index_k = h_index (this, self, i, col), index_m = h_index (this, self, row, col);
        h_ops_tree * op_j = elements[row * nx + col].generateOps_GEMM(&index_m, &elements[row * nx + i], &index_j, &elements[i * nx + col], &index_k);
        op -> setChild(op_j, child_offset[i] + rows + cols + j + 1);
        delete op_j;
      }
    }

    delete[] child_offset;
    return op;
  }

  __host__ h_ops_tree * generateOps_TRSML (const h_index * self, const dev_dense <T> *B, const h_index * index_b) const
  {
    h_ops_tree * op = new h_ops_tree (trsml_d, index_b, self);

    int n = nx > ny ? ny : nx, * child_offset = new int[n], * y = new int[ny];
    child_offset[0] = 0; y[0] = 0;

    for (int i = 1; i < n; i++)
    { child_offset[i] = child_offset[i - 1] + ny - i + 1; }

    for (int i = 1; i < ny; i++)
    { y[i] = y[i - 1] + elements[(i - 1) * nx].getNy(); }

    op -> resizeChildren(child_offset[n - 1] + ny - n + 1);

#pragma omp parallel for num_threads(2)
    for (int i = 0; i < n; i++)
    {
      const h_index index_i = h_index (this, self, i, i), index_bi = h_index (index_b, y[i], 0, index_i.getNy(), index_b -> getNx());
      h_ops_tree * op_i = elements[i * nx + i].generateOps_TRSML(&index_i, B, &index_bi);
      op -> setChild(op_i, child_offset[i]);
      delete op_i;

      for (int j = i + 1; j < ny; j++)
      {
        const h_index index_j = h_index (this, self, j, i), index_bj = h_index (index_b, y[j], 0, index_j.getNy(), index_b -> getNx());
        h_ops_tree * op_j = B -> generateOps_GEMM(&index_bj, &elements[j * nx + i], &index_j, B, &index_bi);
        op -> setChild(op_j, child_offset[i] + j - i);
        delete op_j;
      }
    }

    delete[] child_offset;
    delete[] y;

    return op;
  }

  __host__ h_ops_tree * generateOps_TRSML (const h_index *self, const dev_low_rank <T> *B, const h_index *index_b) const
  {
    h_ops_tree * op = new h_ops_tree (trsml_lr, index_b, self);

    int n = nx > ny ? ny : nx, * child_offset = new int[n], * y = new int[ny];
    child_offset[0] = 0; y[0] = 0;

    for (int i = 1; i < n; i++)
    { child_offset[i] = child_offset[i - 1] + ny - i + 1; }

    for (int i = 1; i < ny; i++)
    { y[i] = y[i - 1] + elements[(i - 1) * nx].getNy(); }

    op -> resizeChildren(child_offset[n - 1] + ny - n + 1);

#pragma omp parallel for num_threads(2)
    for (int i = 0; i < n; i++)
    {
      const h_index index_i = h_index (this, self, i, i), index_bi = h_index (index_b, y[i], 0, index_i.getNy(), index_b -> getNx());
      h_ops_tree * op_i = elements[i * nx + i].generateOps_TRSML(&index_i, B, &index_bi);
      op -> setChild(op_i, child_offset[i]);
      delete op_i;

      for (int j = i + 1; j < ny; j++)
      {
        const h_index index_j = h_index (this, self, j, i), index_bj = h_index (index_b, y[j], 0, index_j.getNy(), index_b -> getNx());
        h_ops_tree * op_j = B -> generateOps_GEMM(&index_bj, &elements[j * nx + i], &index_j, B, &index_bi);
        op -> setChild(op_j, child_offset[i] + j - i);
        delete op_j;
      }
    }

    delete[] child_offset;
    delete[] y;

    return op;
  }

  __host__ h_ops_tree * generateOps_TRSML (const h_index *self, const dev_hierarchical <T> *B, const h_index *index_b) const
  {
    if (ny != B -> ny) 
    { printf("Matrices are partitioned differently in H-H TRSML.\n"); return nullptr; }

    h_ops_tree * op = new h_ops_tree (trsml_d, index_b, self);

    int n = nx > ny ? ny : nx, * child_offset = new int[n];
    child_offset[0] = 0;

    for (int i = 1; i < n; i++)
    { child_offset[i] = child_offset[i - 1] + (B -> nx) * (ny - i + 1); }

    op -> resizeChildren(child_offset[n - 1] + (B -> nx) * (ny - n + 1));

#pragma omp parallel for num_threads(2)
    for (int i = 0; i < n; i++)
    {
      const h_index index_i = h_index (this, self, i, i);

      for (int j = 0; j < B -> nx; j++)
      {
        const h_index index_bj = h_index (B, index_b, i, j);

        h_ops_tree * op_j = elements[i * nx + i].generateOps_TRSML(&index_i, &(B -> elements)[i * (B -> nx) + j], &index_bj);
        op -> setChild(op_j, child_offset[i] + j);
        delete op_j;

        for (int k = i + 1; k < ny; k++)
        {
          const h_index index_k = h_index (this, self, k, i), index_bk = h_index (B, index_b, k, j);
          h_ops_tree * op_k = (B -> elements[k * (B -> nx) + j]).generateOps_GEMM(&index_bk, &elements[k * nx + i], &index_k, &(B -> elements)[i * (B -> nx) + j], &index_bj);
          op -> setChild(op_k, child_offset[i] + (k - i) * B -> nx + j);
          delete op_k;
        }
      }
    }

    delete[] child_offset;
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
    h_ops_tree * op = new h_ops_tree (trsmr_d, index_b, self);

    int n = nx > ny ? ny : nx, * child_offset = new int[n], * x = new int[nx];
    child_offset[0] = 0; x[0] = 0;

    for (int i = 1; i < n; i++)
    { child_offset[i] = child_offset[i - 1] + nx - i + 1; }

    for (int i = 1; i < nx; i++)
    { x[i] = x[i - 1] + elements[i - 1].getNx(); }

    op -> resizeChildren(child_offset[n - 1] + nx - n + 1);

#pragma omp parallel for num_threads(2)
    for (int i = 0; i < n; i++)
    {
      const h_index index_i = h_index (this, self, i, i), index_bi = h_index (index_b, 0, x[i], index_b -> getNy(), index_i.getNx());
      h_ops_tree * op_i = elements[i * nx + i].generateOps_TRSMR(&index_i, B, &index_bi);
      op -> setChild(op_i, child_offset[i]);
      delete op_i;

      for (int j = i + 1; j < nx; j++)
      {
        const h_index index_j = h_index (this, self, i, j), index_bj = h_index (index_b, 0, x[j], index_b -> getNy(), index_j.getNx());
        h_ops_tree * op_j = B -> generateOps_GEMM(&index_bj, &elements[j * nx + i], &index_j, B, &index_bi);
        op -> setChild(op_j, child_offset[i] + j - i);
        delete op_j;
      }
    }

    delete[] child_offset;
    delete[] x;

    return op;
  }

  __host__ h_ops_tree * generateOps_TRSMR (const h_index *self, const dev_low_rank <T> *B, const h_index *index_b) const
  {
    h_ops_tree * op = new h_ops_tree (trsmr_lr, index_b, self);

    int n = nx > ny ? ny : nx, * child_offset = new int[n], * x = new int[nx];
    child_offset[0] = 0; x[0] = 0;

    for (int i = 1; i < n; i++)
    { child_offset[i] = child_offset[i - 1] + nx - i + 1; }

    for (int i = 1; i < nx; i++)
    { x[i] = x[i - 1] + elements[i - 1].getNx(); }

    op -> resizeChildren(child_offset[n - 1] + nx - n + 1);

#pragma omp parallel for num_threads(2)
    for (int i = 0; i < n; i++)
    {
      const h_index index_i = h_index (this, self, i, i), index_bi = h_index (index_b, 0, x[i], index_b -> getNy(), index_i.getNx());
      h_ops_tree * op_i = elements[i * nx + i].generateOps_TRSMR(&index_i, B, &index_bi);
      op -> setChild(op_i, child_offset[i]);
      delete op_i;

      for (int j = i + 1; j < nx; j++)
      {
        const h_index index_j = h_index (this, self, i, j), index_bj = h_index (index_b, 0, x[j], index_b -> getNy(), index_j.getNx());
        h_ops_tree * op_j = B -> generateOps_GEMM(&index_bj, &elements[j * nx + i], &index_j, B, &index_bi);
        op -> setChild(op_j, child_offset[i] + j - i);
        delete op_j;
      }
    }

    delete[] child_offset;
    delete[] x;

    return op;
  }

  __host__ h_ops_tree * generateOps_TRSMR (const h_index *self, const dev_hierarchical <T> *B, const h_index *index_b) const
  {
    if (nx != B -> nx)
    { printf("Matrices are partitioned differently in H-H TRSMR.\n"); return nullptr; }

    h_ops_tree * op = new h_ops_tree (trsml_d, index_b, self);

    int n = nx > ny ? ny : nx, * child_offset = new int[n];
    child_offset[0] = 0;

    for (int i = 1; i < n; i++)
    { child_offset[i] = child_offset[i - 1] + (B -> ny) * (nx - i + 1); }

    op -> resizeChildren(child_offset[n - 1] + (B -> ny) * (nx - n + 1));

#pragma omp parallel for num_threads(2)
    for (int i = 0; i < n; i++)
    {
      const h_index index_i = h_index (this, self, i, i);

      for (int j = 0; j < B -> ny; j++)
      {
        const h_index index_bj = h_index (B, index_b, j, i);

        h_ops_tree * op_j = elements[i * nx + i].generateOps_TRSMR(&index_i, &(B -> elements)[j * (B -> nx) + i], &index_bj);
        op -> setChild(op_j, child_offset[i] + j);
        delete op_j;

        for (int k = i + 1; k < nx; k++)
        {
          const h_index index_k = h_index (this, self, i, k), index_bk = h_index (B, index_b, j, k);
          h_ops_tree * op_k = (B -> elements[j * (B -> nx) + k]).generateOps_GEMM(&index_bk, &(B -> elements)[j * (B -> nx) + i], &index_bj, &elements[i * nx + k], &index_k);
          op -> setChild(op_k, child_offset[i] + (k - i) * B -> ny + j);
          delete op_k;
        }
      }
    }

    delete[] child_offset;
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

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_dense <T> *A, const h_index *index_a, const dev_dense <T> *B, const h_index *index_b) const
  {
    h_ops_tree * op = new h_ops_tree (gemm_d_d_d, self, index_a, index_b);
    op -> resizeChildren(nx * ny);

    int * x = new int[nx], * y = new int[ny], k = A -> getNx();
    x[0] = 0; y[0] = 0; k = (B -> getNy() > k) ? k : B -> getNy();

    for (int i = 1; i < nx; i++)
    { x[i] = x[i - 1] + elements[i - 1].getNx(); }

    for (int i = 1; i < ny; i++)
    { y[i] = y[i - 1] + elements[(i - 1) * nx].getNy(); }

#pragma omp parallel for num_threads(2)
    for (int i = 0; i < ny * nx; i++)
    {
      const int row = i / nx, col = i - row * nx;
      const h_index index_m = h_index (this, self, row, col), index_ai = h_index (index_a, y[row], 0, index_m.getNy(), k), index_bj = h_index (index_b, 0, x[col], k, index_m.getNx());
      h_ops_tree * op_i = elements[i].generateOps_GEMM(&index_m, A, &index_ai, B, &index_bj);
      op -> setChild(op_i, i);
      delete op_i;
    }

    delete[] x;
    delete[] y;
    return op;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_low_rank <T> *A, const h_index *index_a, const dev_dense <T> *B, const h_index *index_b) const
  {
    h_ops_tree * op = new h_ops_tree (gemm_d_lr_d, self, index_a, index_b);
    op -> resizeChildren(nx * ny);

    int * x = new int[nx], * y = new int[ny], k = A -> getNx();
    x[0] = 0; y[0] = 0; k = (B -> getNy() > k) ? k : B -> getNy();

    for (int i = 1; i < nx; i++)
    { x[i] = x[i - 1] + elements[i - 1].getNx(); }

    for (int i = 1; i < ny; i++)
    { y[i] = y[i - 1] + elements[(i - 1) * nx].getNy(); }

#pragma omp parallel for num_threads(2)
    for (int i = 0; i < ny * nx; i++)
    {
      const int row = i / nx, col = i - row * nx;
      const h_index index_m = h_index (this, self, row, col), index_ai = h_index (index_a, y[row], 0, index_m.getNy(), k), index_bj = h_index (index_b, 0, x[col], k, index_m.getNx());
      h_ops_tree * op_i = elements[i].generateOps_GEMM(&index_m, A, &index_ai, B, &index_bj);
      op -> setChild(op_i, i);
      delete op_i;
    }

    delete[] x;
    delete[] y;
    return op;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_hierarchical <T> *A, const h_index *index_a, const dev_dense <T> *B, const h_index *index_b) const
  {
    if (ny != A -> ny)
    { printf("Matrices are partitioned differently in H-H.D GEMM.\n"); return nullptr; }

    h_ops_tree * op = new h_ops_tree (gemm_d_d_d, self, index_a, index_b);
    op -> resizeChildren(nx * ny * A -> nx);

    int * x = new int[nx], * y = new int[A -> nx];
    x[0] = 0; y[0] = 0;

    for (int i = 1; i < nx; i++)
    { x[i] = x[i - 1] + elements[i - 1].getNx(); }

    for (int i = 1; i < A -> nx; i++)
    { y[i] = y[i - 1] + (A -> elements)[i - 1].getNx(); }

#pragma omp parallel for num_threads(2)
    for (int i = 0; i < ny * nx; i++)
    {
      const int row = i / nx, col = i - row * nx;
      const h_index index_m = h_index (this, self, row, col);
      for (int k = 0; k < A -> nx; k++)
      {
        const h_index index_ak = h_index (A, index_a, row, k), index_bk = h_index (index_b, y[k], x[col], index_ak.getNx(), index_m.getNx());
        h_ops_tree * op_k = elements[i].generateOps_GEMM(&index_m, &(A -> elements[row * (A -> nx) + k]), &index_ak, B, &index_bk);
        op -> setChild(op_k, i * (A -> nx) + k);
        delete op_k;
      }
    }

    delete[] x;
    delete[] y;
    return op;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_h_element <T> *A, const h_index *index_a, const dev_dense <T> *B, const h_index *index_b) const
  {
    const dev_hierarchical <T> *h_a = A -> getElementHierarchical();
    const dev_low_rank <T> *lr_a = A -> getElementLowRank();
    const dev_dense <T> *d_a = A -> getElementDense();

    if (d_a != nullptr)
    { return generateOps_GEMM (self, d_a, index_a, B, index_b); }
    if (lr_a != nullptr)
    { return generateOps_GEMM (self, lr_a, index_a, B, index_b); }
    if (h_a != nullptr)
    { return generateOps_GEMM (self, h_a, index_a, B, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_dense <T> *A, const h_index *index_a, const dev_low_rank <T> *B, const h_index *index_b) const
  {
    h_ops_tree * op = new h_ops_tree (gemm_d_d_lr, self, index_a, index_b);
    op -> resizeChildren(nx * ny);

    int * x = new int[nx], * y = new int[ny], k = A -> getNx();
    x[0] = 0; y[0] = 0; k = (B -> getNy() > k) ? k : B -> getNy();

    for (int i = 1; i < nx; i++)
    { x[i] = x[i - 1] + elements[i - 1].getNx(); }

    for (int i = 1; i < ny; i++)
    { y[i] = y[i - 1] + elements[(i - 1) * nx].getNy(); }

#pragma omp parallel for num_threads(2)
    for (int i = 0; i < ny * nx; i++)
    {
      const int row = i / nx, col = i - row * nx;
      const h_index index_m = h_index (this, self, row, col), index_ai = h_index (index_a, y[row], 0, index_m.getNy(), k), index_bj = h_index (index_b, 0, x[col], k, index_m.getNx());
      h_ops_tree * op_i = elements[i].generateOps_GEMM(&index_m, A, &index_ai, B, &index_bj);
      op -> setChild(op_i, i);
      delete op_i;
    }

    delete[] x;
    delete[] y;
    return op;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_low_rank <T> *A, const h_index *index_a, const dev_low_rank <T> *B, const h_index *index_b) const
  {
    h_ops_tree * op = new h_ops_tree (gemm_d_lr_lr, self, index_a, index_b);
    op -> resizeChildren(nx * ny);

    int * x = new int[nx], * y = new int[ny], k = A -> getNx();
    x[0] = 0; y[0] = 0; k = (B -> getNy() > k) ? k : B -> getNy();

    for (int i = 1; i < nx; i++)
    { x[i] = x[i - 1] + elements[i - 1].getNx(); }

    for (int i = 1; i < ny; i++)
    { y[i] = y[i - 1] + elements[(i - 1) * nx].getNy(); }

#pragma omp parallel for num_threads(2)
    for (int i = 0; i < ny * nx; i++)
    {
      const int row = i / nx, col = i - row * nx;
      const h_index index_m = h_index (this, self, row, col), index_ai = h_index (index_a, y[row], 0, index_m.getNy(), k), index_bj = h_index (index_b, 0, x[col], k, index_m.getNx());
      h_ops_tree * op_i = elements[i].generateOps_GEMM(&index_m, A, &index_ai, B, &index_bj);
      op -> setChild(op_i, i);
      delete op_i;
    }

    delete[] x;
    delete[] y;
    return op;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_hierarchical <T> *A, const h_index *index_a, const dev_low_rank <T> *B, const h_index *index_b) const
  {
    if (ny != A -> ny)
    { printf("Matrices are partitioned differently in H-H.LR GEMM.\n"); return nullptr; }

    h_ops_tree * op = new h_ops_tree (gemm_d_d_lr, self, index_a, index_b);
    op -> resizeChildren(nx * ny * A -> nx);

    int * x = new int[nx], * y = new int[A -> nx];
    x[0] = 0; y[0] = 0;

    for (int i = 1; i < nx; i++)
    { x[i] = x[i - 1] + elements[i - 1].getNx(); }

    for (int i = 1; i < A -> nx; i++)
    { y[i] = y[i - 1] + (A -> elements)[i - 1].getNx(); }

#pragma omp parallel for num_threads(2)
    for (int i = 0; i < ny * nx; i++)
    {
      const int row = i / nx, col = i - row * nx;
      const h_index index_m = h_index (this, self, row, col);
      for (int k = 0; k < A -> nx; k++)
      {
        const h_index index_ak = h_index (A, index_a, row, k), index_bk = h_index (index_b, y[k], x[col], index_ak.getNx(), index_m.getNx());
        h_ops_tree * op_k = elements[i].generateOps_GEMM(&index_m, &(A -> elements[row * (A -> nx) + k]), &index_ak, B, &index_bk);
        op -> setChild(op_k, i * (A -> nx) + k);
        delete op_k;
      }
    }

    delete[] x;
    delete[] y;
    return op;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_h_element <T> *A, const h_index *index_a, const dev_low_rank <T> *B, const h_index *index_b) const
  {
    const dev_hierarchical <T> *h_a = A -> getElementHierarchical();
    const dev_low_rank <T> *lr_a = A -> getElementLowRank();
    const dev_dense <T> *d_a = A -> getElementDense();

    if (d_a != nullptr)
    { return generateOps_GEMM (self, d_a, index_a, B, index_b); }
    if (lr_a != nullptr)
    { return generateOps_GEMM (self, lr_a, index_a, B, index_b); }
    if (h_a != nullptr)
    { return generateOps_GEMM (self, h_a, index_a, B, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_dense <T> *A, const h_index *index_a, const dev_hierarchical <T> *B, const h_index *index_b) const
  {
    if (nx != B -> nx)
    { printf("Matrices are partitioned differently in H-D.H GEMM.\n"); return nullptr; }

    h_ops_tree * op = new h_ops_tree (gemm_d_d_d, self, index_a, index_b);
    op -> resizeChildren (nx * ny * B -> ny);

    int * x = new int[B -> ny], * y = new int[ny];
    x[0] = 0; y[0] = 0;

    for (int i = 1; i < B -> ny; i++)
    { x[i] = x[i - 1] + (B -> elements)[(i - 1) * (B -> nx)].getNy(); }

    for (int i = 1; i < ny; i++)
    { y[i] = y[i - 1] + elements[(i - 1) * nx].getNy(); }

#pragma omp parallel for num_threads(2)
    for (int i = 0; i < ny * nx; i++)
    {
      const int row = i / nx, col = i - row * nx;
      const h_index index_m = h_index (this, self, row, col);
      for (int k = 0; k < B -> ny; k++)
      {
        const h_index index_bk = h_index (B, index_b, k, col), index_ak = h_index (index_a, y[row], x[k], index_m.getNy(), index_bk.getNy());
        h_ops_tree * op_k = elements[i].generateOps_GEMM(&index_m, A, &index_ak, &(B -> elements[k * (B -> nx) + col]), &index_bk);
        op -> setChild(op_k, i * (B -> ny) + k);
        delete op_k;
      }
    }

    delete[] x;
    delete[] y;

    return op;  
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_low_rank <T> *A, const h_index *index_a, const dev_hierarchical <T> *B, const h_index *index_b) const
  {
    if (nx != B -> nx)
    { printf("Matrices are partitioned differently in H-LR.H GEMM.\n"); return nullptr; }

    h_ops_tree * op = new h_ops_tree (gemm_d_lr_d, self, index_a, index_b);
    op -> resizeChildren (nx * ny * B -> ny);

    int * x = new int[B -> ny], * y = new int[ny];
    x[0] = 0; y[0] = 0;

    for (int i = 1; i < B -> ny; i++)
    { x[i] = x[i - 1] + (B -> elements)[(i - 1) * (B -> nx)].getNy(); }

    for (int i = 1; i < ny; i++)
    { y[i] = y[i - 1] + elements[(i - 1) * nx].getNy(); }

#pragma omp parallel for num_threads(2)
    for (int i = 0; i < ny * nx; i++)
    {
      const int row = i / nx, col = i - row * nx;
      const h_index index_m = h_index (this, self, row, col);
      for (int k = 0; k < B -> ny; k++)
      {
        const h_index index_bk = h_index (B, index_b, k, col), index_ak = h_index (index_a, y[row], x[k], index_m.getNy(), index_bk.getNy());
        h_ops_tree * op_k = elements[i].generateOps_GEMM(&index_m, A, &index_ak, &(B -> elements[k * (B -> nx) + col]), &index_bk);
        op -> setChild(op_k, i * (B -> ny) + k);
        delete op_k;
      }
    }

    delete[] x;
    delete[] y;

    return op;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_hierarchical <T> *A, const h_index *index_a, const dev_hierarchical <T> *B, const h_index *index_b) const
  {
    if (ny != A -> ny || nx != B -> nx || A -> nx != B -> ny)
    { printf("Matrices are partitioned differently in H-H.H GEMM.\n"); return nullptr; }

    h_ops_tree * op = new h_ops_tree (gemm_d_d_d, self, index_a, index_b);
    op -> resizeChildren(nx * ny * A -> nx);

#pragma omp parallel for num_threads(2)
    for (int i = 0; i < ny * nx; i++)
    {
      const int row = i / nx, col = i - row * nx;
      const h_index index_m = h_index (this, self, row, col);
      for (int k = 0; k < A -> nx; k++)
      {
        const h_index index_ak = h_index (A, index_a, row, k), index_bk = h_index (B, index_b, k, col);
        h_ops_tree * op_k = elements[i].generateOps_GEMM(&index_m, &(A -> elements)[row * (A -> nx) + k], &index_ak, &(B -> elements)[k * (B -> nx) + col], &index_bk);
        op -> setChild(op_k, i * (A -> nx) + k);
        delete op_k;
      }
    }
    return op;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_h_element <T> *A, const h_index *index_a, const dev_hierarchical <T> *B, const h_index *index_b) const
  {
    const dev_hierarchical <T> *h_a = A -> getElementHierarchical();
    const dev_low_rank <T> *lr_a = A -> getElementLowRank();
    const dev_dense <T> *d_a = A -> getElementDense();

    if (d_a != nullptr)
    { return generateOps_GEMM (self, d_a, index_a, B, index_b); }
    if (lr_a != nullptr)
    { return generateOps_GEMM (self, lr_a, index_a, B, index_b); }
    if (h_a != nullptr)
    { return generateOps_GEMM (self, h_a, index_a, B, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_dense <T> *A, const h_index *index_a, const dev_h_element <T> *B, const h_index *index_b) const
  {
    const dev_hierarchical <T> *h_b = B -> getElementHierarchical();
    const dev_low_rank <T> *lr_b = B -> getElementLowRank();
    const dev_dense <T> *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, d_b, index_b); }
    if (lr_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, lr_b, index_b); }
    if (h_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, h_b, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_low_rank <T> *A, const h_index *index_a, const dev_h_element <T> *B, const h_index *index_b) const
  {
    const dev_hierarchical <T> *h_b = B -> getElementHierarchical();
    const dev_low_rank <T> *lr_b = B -> getElementLowRank();
    const dev_dense <T> *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, d_b, index_b); }
    if (lr_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, lr_b, index_b); }
    if (h_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, h_b, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_hierarchical <T> *A, const h_index *index_a, const dev_h_element <T> *B, const h_index *index_b) const
  {
    const dev_hierarchical <T> *h_b = B -> getElementHierarchical();
    const dev_low_rank <T> *lr_b = B -> getElementLowRank();
    const dev_dense <T> *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, d_b, index_b); }
    if (lr_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, lr_b, index_b); }
    if (h_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, h_b, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_h_element <T> *A, const h_index *index_a, const dev_h_element <T> *B, const h_index *index_b) const
  {
    const dev_hierarchical <T> *h_b = B -> getElementHierarchical();
    const dev_low_rank <T> *lr_b = B -> getElementLowRank();
    const dev_dense <T> *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, d_b, index_b); }
    if (lr_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, lr_b, index_b); }
    if (h_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, h_b, index_b); }

    return nullptr;
  }

  __host__ void print(const h_index * index_in) const
  {
    for (int i = 0; i < ny * nx; i++)
    {
      const int row = i / nx, col = i - row * nx;
      const h_index * i_index = new h_index(this, index_in, row, col);
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

  __host__ void loadTestMatrix (const int levels, const int dim, const int block_size, const int admis, const int x_start = 0, const int y_start = 0)
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
          e -> loadTestMatrix(levels - 1, dim, block_size, admis, x_offset, y_offset); 
          setElement(e, hierarchical, x, y);
          x_offset += e -> getNx_abs();
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
            setElement(e, low_rank, x, y);
            x_offset += e -> getNx();
          }
        }
      }
      y_offset += elements[y * nx].getNy();
    }

    updateOffsets();

  }


};

#endif