
#ifndef _DEV_DENSE_CUH
#define _DEV_DENSE_CUH

#include <pspl.cuh>

template <class T> class dev_dense 
{
private:
  int device_id;

  int nx;
  int ny;
  int ld;

  T * elements;

  bool pivoted;
  int * pivot;

public:

  __host__ dev_dense (const int nx_in = 0, const int ny_in = 0, const int ld_in = 0, const int device_id_in = 0, const bool alloc_pivot = false)
  {
    nx = nx_in;
    ny = ny_in;
    ld = (nx > ld_in) ? nx : ld_in;

    if (device_id_in >= 0 && cudaSetDevice(device_id_in) == cudaSuccess)
    { 
      device_id = device_id_in;

      if (cudaMallocManaged(&elements, ld * ny * sizeof(T), cudaMemAttachGlobal) == cudaSuccess)
      { cudaMemset(elements, 0, ld * ny * sizeof(T)); }
      else
      { elements = nullptr; }
    
      if (alloc_pivot && cudaMallocManaged(&pivot, ny * sizeof(int), cudaMemAttachGlobal) == cudaSuccess)
      { cudaMemset(pivot, 0, ny * sizeof(int)); pivoted = true; }
      else
      { pivot = nullptr; pivoted = false; }
    }
    else
    { 
      device_id = -1;
      elements = nullptr;
      pivoted = false;
      pivot = nullptr;
    }
  }

  __host__ ~dev_dense ()
  {
    cudaFree(elements);
    if (pivoted) { cudaFree(pivot); }
  }

  __host__ inline int getNx () const 
  { return nx; }

  __host__ inline int getNy () const 
  { return ny; }

  __host__ inline int getLd () const 
  { return ld; }

  __host__ inline T * getElements (const int offset = 0) const 
  { return &elements[offset]; }

  __host__ inline int * getPivot (const int offset = 0) const 
  { return pivoted ? &pivot[offset / ld] : nullptr; }

  __host__ cudaError_t resize (const int ld_in, const int ny_in)
  {
    cudaError_t error = resizeColumn(ld_in);
    return error == cudaSuccess ? resizeRow(ny_in) : error;
  }

  __host__ cudaError_t resizeColumn (const int ld_in)
  {
    if (ld_in > 0 && ld_in != ld)
    {
      T * e = nullptr;
      cudaError_t error = cudaMallocManaged (&e, ld_in * ny * sizeof(T), cudaMemAttachGlobal);
      if (error != cudaSuccess) { return error; }

      for (int y = 0; y < ny; y++) for (int x = 0; x < nx && x < ld_in; x++)
      { e[y * ld_in + x] = elements[y * ld + x]; }

      cudaFree(elements);
      ld = ld_in;
      nx = (nx > ld) ? ld : nx;
      elements = e;
    }
    return cudaSuccess;
  }

  __host__ cudaError_t resizeRow (const int ny_in)
  {
    if (ny_in > 0 && ny_in != ny)
    {
      T * e = nullptr;
      cudaError_t error = cudaMallocManaged (&e, ld * ny_in * sizeof(T), cudaMemAttachGlobal);
      if (error != cudaSuccess) { return error; }

      for (int y = 0; y < ny_in && y < ny; y++) for (int x = 0; x < nx; x++)
      { e[y * ld + x] = elements[y * ld + x]; }

      cudaFree(elements);
      elements = e;

      if (pivoted)
      {
        int * p = nullptr;
        error = cudaMallocManaged (&p, ny_in * sizeof(int), cudaMemAttachGlobal);
        if (error != cudaSuccess) { return error; }

        for (int y = 0; y < ny_in && y < ny; y++)
        { p[y] = pivot[y]; }

        cudaFree(pivot);
        pivot = p;
      }

      ny = ny_in;
    }
    return cudaSuccess;
  }

  __host__ static h_ops_tree * generateOps_GETRF (const h_index * self, dev_temp * tmp_mngr)
  { 
    return new h_ops_tree (getrf, self); 
  }

  __host__ static h_ops_tree * generateOps_TRSML (const h_index * self, const dev_dense <T> * B, const h_index * index_b, dev_temp * tmp_mngr)
  {
    return new h_ops_tree (trsml, index_b, self);
  }

  __host__ static h_ops_tree * generateOps_TRSML (const h_index * self, const dev_low_rank <T> * B, const h_index * index_b, dev_temp * tmp_mngr)
  {
    h_index index_lr = h_index (index_b); index_lr.setU();
    return new h_ops_tree (trsml, &index_lr, self);
  }

  __host__ static h_ops_tree * generateOps_TRSML (const h_index * self, const dev_hierarchical <T> * B, const h_index * index_b, dev_temp * tmp_mngr)
  {
    return nullptr;
  }

  __host__ static h_ops_tree * generateOps_TRSML (const h_index * self, const dev_h_element <T> * B, const h_index * index_b, dev_temp * tmp_mngr)
  {
    const dev_hierarchical <T> *h_b = B -> getElementHierarchical();
    const dev_low_rank <T> *lr_b = B -> getElementLowRank();
    const dev_dense <T> *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_TRSML (self, d_b, index_b, tmp_mngr); }
    if (lr_b != nullptr)
    { return generateOps_TRSML (self, lr_b, index_b, tmp_mngr); }
    if (h_b != nullptr)
    { return generateOps_TRSML (self, h_b, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ static h_ops_tree * generateOps_TRSMR (const h_index * self, const dev_dense <T> * B, const h_index * index_b, dev_temp * tmp_mngr)
  {
    return new h_ops_tree (trsmr, index_b, self);
  }

  __host__ static h_ops_tree * generateOps_TRSMR (const h_index * self, const dev_low_rank <T> * B, const h_index * index_b, dev_temp * tmp_mngr)
  {
    h_index index_lr = h_index (index_b); index_lr.setVT();
    return new h_ops_tree (trsmr, &index_lr, self);
  }

  __host__ static h_ops_tree * generateOps_TRSMR (const h_index * self, const dev_hierarchical <T> * B, const h_index * index_b, dev_temp * tmp_mngr)
  {
    return nullptr;
  }

  __host__ static h_ops_tree * generateOps_TRSMR (const h_index * self, const dev_h_element <T> * B, const h_index * index_b, dev_temp * tmp_mngr)
  {
    const dev_hierarchical <T> *h_b = B -> getElementHierarchical();
    const dev_low_rank <T> *lr_b = B -> getElementLowRank();
    const dev_dense <T> *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_TRSMR (self, d_b, index_b, tmp_mngr); }
    if (lr_b != nullptr)
    { return generateOps_TRSMR (self, lr_b, index_b, tmp_mngr); }
    if (h_b != nullptr)
    { return generateOps_TRSMR (self, h_b, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ static h_ops_tree * generateOps_ACCM (const h_index * self, const h_index * index_tmp_lr)
  {
    return new h_ops_tree (accum, self, index_tmp_lr); 
  }

  __host__ static h_ops_tree * generateOps_GEMM (const h_index * self, const dev_dense <T> * A, const h_index * index_a, const dev_dense <T> * B, const h_index * index_b, dev_temp * tmp_mngr)
  {
    return new h_ops_tree (gemm, self, index_a, index_b); 
  }

  __host__ static h_ops_tree * generateOps_GEMM (const h_index * self, const dev_low_rank <T> * A, const h_index * index_a, const dev_dense <T> * B, const h_index * index_b, dev_temp * tmp_mngr)
  {
    return new h_ops_tree (gemm, self, index_a, index_b); 
  }

  __host__ static h_ops_tree * generateOps_GEMM (const h_index * self, const dev_hierarchical <T> * A, const h_index * index_a, const dev_dense <T> * B, const h_index * index_b, dev_temp * tmp_mngr)
  {
    h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b);
    op -> resizeChildren(A -> getNx_blocks() * A -> getNy_blocks());

    int * y, * k, x = self -> getNx(index_b -> getNx());
    A -> getOffsets_y(&y);
    A -> getOffsets_x(&k);

#pragma omp parallel for if (omp_in_parallel() == 0)
    for (int i = 0; i < A -> getNx_blocks() * A -> getNy_blocks(); i++)
    {
      const int row = i / (A -> getNx_blocks()), col = i - row * (A -> getNx_blocks());
      const h_index index_ai = h_index (A, index_a, row, col), index_m = h_index (self, y[row], 0, index_ai.getNy(), x), index_bj = h_index (index_b, k[col], 0, index_ai.getNx(), x);
      h_ops_tree * op_i = generateOps_GEMM(&index_m, A -> getElement_blocks(row, col), &index_ai, B, &index_bj, tmp_mngr);
      op -> setChild(op_i, i);
      delete op_i;
    }

    delete[] y;
    delete[] k;
    return op;
  }

  __host__ static h_ops_tree * generateOps_GEMM (const h_index * self, const dev_h_element <T> * A, const h_index * index_a, const dev_dense <T> * B, const h_index * index_b, dev_temp * tmp_mngr)
  {
    const dev_hierarchical <T> *h_a = A -> getElementHierarchical();
    const dev_low_rank <T> *lr_a = A -> getElementLowRank();
    const dev_dense <T> *d_a = A -> getElementDense();

    if (d_a != nullptr)
    { return generateOps_GEMM (self, d_a, index_a, B, index_b, tmp_mngr); }
    if (lr_a != nullptr)
    { return generateOps_GEMM (self, lr_a, index_a, B, index_b, tmp_mngr); }
    if (h_a != nullptr)
    { return generateOps_GEMM (self, h_a, index_a, B, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ static h_ops_tree * generateOps_GEMM (const h_index * self, const dev_dense <T> * A, const h_index * index_a, const dev_low_rank <T> * B, const h_index * index_b, dev_temp * tmp_mngr)
  {
    return new h_ops_tree (gemm, self, index_a, index_b);
  }

  __host__ static h_ops_tree * generateOps_GEMM (const h_index * self, const dev_low_rank <T> * A, const h_index * index_a, const dev_low_rank <T> * B, const h_index * index_b, dev_temp * tmp_mngr)
  {
    return new h_ops_tree (gemm, self, index_a, index_b);
  }

  __host__ static h_ops_tree * generateOps_GEMM (const h_index * self, const dev_hierarchical <T> * A, const h_index * index_a, const dev_low_rank <T> * B, const h_index * index_b, dev_temp * tmp_mngr)
  {
    h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b);
    op -> resizeChildren(A -> getNx_blocks() * A -> getNy_blocks());

    int * y, * k, x = self -> getNx(index_b -> getNx());
    A -> getOffsets_y(&y);
    A -> getOffsets_x(&k);

#pragma omp parallel for if (omp_in_parallel() == 0)
    for (int i = 0; i < A -> getNx_blocks() * A -> getNy_blocks(); i++)
    {
      const int row = i / (A -> getNx_blocks()), col = i - row * (A -> getNx_blocks());
      const h_index index_ai = h_index (A, index_a, row, col), index_m = h_index (self, y[row], 0, index_ai.getNy(), x), index_bj = h_index (index_b, k[col], 0, index_ai.getNx(), x);
      h_ops_tree * op_i = generateOps_GEMM(&index_m, A -> getElement_blocks(row, col), &index_ai, B, &index_bj, tmp_mngr);
      op -> setChild(op_i, i);
      delete op_i;
    }

    delete[] y;
    delete[] k;
    return op;
  }

  __host__ static h_ops_tree * generateOps_GEMM (const h_index * self, const dev_h_element <T> * A, const h_index * index_a, const dev_low_rank <T> * B, const h_index * index_b, dev_temp * tmp_mngr)
  {
    const dev_hierarchical <T> *h_a = A -> getElementHierarchical();
    const dev_low_rank <T> *lr_a = A -> getElementLowRank();
    const dev_dense <T> *d_a = A -> getElementDense();

    if (d_a != nullptr)
    { return generateOps_GEMM (self, d_a, index_a, B, index_b, tmp_mngr); }
    if (lr_a != nullptr)
    { return generateOps_GEMM (self, lr_a, index_a, B, index_b, tmp_mngr); }
    if (h_a != nullptr)
    { return generateOps_GEMM (self, h_a, index_a, B, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ static h_ops_tree * generateOps_GEMM (const h_index * self, const dev_dense <T> * A, const h_index * index_a, const dev_hierarchical <T> * B, const h_index * index_b, dev_temp * tmp_mngr)
  {
    h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b);
    op -> resizeChildren(B -> getNx_blocks() * B -> getNy_blocks());

    int * x, * k, y = self -> getNy(index_a -> getNy());
    B -> getOffsets_y(&k);
    B -> getOffsets_x(&x);

#pragma omp parallel for if (omp_in_parallel() == 0)
    for (int i = 0; i < B -> getNx_blocks() * B -> getNy_blocks(); i++)
    {
      const int row = i / (B -> getNx_blocks()), col = i - row * (B -> getNx_blocks());
      const h_index index_bj = h_index (B, index_b, row, col), index_m = h_index (self, 0, x[col], y, index_bj.getNx()), index_ai = h_index (index_a, 0, k[row], y, index_bj.getNy());
      h_ops_tree * op_i = generateOps_GEMM(&index_m, A, &index_ai, B -> getElement_blocks(row, col), &index_bj, tmp_mngr);
      op -> setChild(op_i, i);
      delete op_i;
    }

    delete[] x;
    delete[] k;
    return op;
  }

  __host__ static h_ops_tree * generateOps_GEMM (const h_index * self, const dev_low_rank <T> * A, const h_index * index_a, const dev_hierarchical <T> * B, const h_index * index_b, dev_temp * tmp_mngr)
  {
    h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b);
    op -> resizeChildren(B -> getNx_blocks() * B -> getNy_blocks());

    int * x, * k, y = self -> getNy(index_a -> getNy());
    B -> getOffsets_y(&k);
    B -> getOffsets_x(&x);

#pragma omp parallel for if (omp_in_parallel() == 0)
    for (int i = 0; i < B -> getNx_blocks() * B -> getNy_blocks(); i++)
    {
      const int row = i / (B -> getNx_blocks()), col = i - row * (B -> getNx_blocks());
      const h_index index_bj = h_index (B, index_b, row, col), index_m = h_index (self, 0, x[col], y, index_bj.getNx()), index_ai = h_index (index_a, 0, k[row], y, index_bj.getNy());
      h_ops_tree * op_i = generateOps_GEMM(&index_m, A, &index_ai, B -> getElement_blocks(row, col), &index_bj, tmp_mngr);
      op -> setChild(op_i, i);
      delete op_i;
    }

    delete[] x;
    delete[] k;
    return op;
  }

  __host__ static h_ops_tree * generateOps_GEMM (const h_index * self, const dev_hierarchical <T> * A, const h_index * index_a, const dev_hierarchical <T> * B, const h_index * index_b, dev_temp * tmp_mngr)
  {
    const int n_k = A -> getNx_blocks(); if (n_k != B -> getNy_blocks())
    { printf("Matrices are partitioned differently in D.H-H GEMM.\n"); return nullptr; }

    h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b);
    const int n_n = B -> getNx_blocks(), n_mn = n_n * A -> getNy_blocks(), n_mnk = n_mn * n_k;
    int * x, * y;
    A -> getOffsets_y(&y);
    B -> getOffsets_x(&x);

    op -> resizeChildren(n_mnk);

#pragma omp parallel for if (omp_in_parallel() == 0)
    for (int i = 0; i < n_mnk; i++)
    {
      const int k = i / n_mn, crd = i - k * n_mn, row = crd / n_n, col = crd - row * n_n;

      const h_index index_ai = h_index (A, index_a, row, k), index_bj = h_index (B, index_b, k, col);
      const h_index index_m = h_index (self, y[row], x[col], index_ai.getNy(), index_bj.getNx());
      h_ops_tree * op_k = generateOps_GEMM (&index_m, A -> getElement_blocks(row, k), &index_ai, B -> getElement_blocks(k, col), &index_bj, tmp_mngr);
      op -> setChild(op_k, i);
      delete op_k;
    }

    delete[] x;
    delete[] y;
    return op;
  }

  __host__ static h_ops_tree * generateOps_GEMM (const h_index * self, const dev_h_element <T> * A, const h_index * index_a, const dev_hierarchical <T> * B, const h_index * index_b, dev_temp * tmp_mngr)
  {
    const dev_hierarchical <T> *h_a = A -> getElementHierarchical();
    const dev_low_rank <T> *lr_a = A -> getElementLowRank();
    const dev_dense <T> *d_a = A -> getElementDense();

    if (d_a != nullptr)
    { return generateOps_GEMM (self, d_a, index_a, B, index_b, tmp_mngr); }
    if (lr_a != nullptr)
    { return generateOps_GEMM (self, lr_a, index_a, B, index_b, tmp_mngr); }
    if (h_a != nullptr)
    { return generateOps_GEMM (self, h_a, index_a, B, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ static h_ops_tree * generateOps_GEMM (const h_index * self, const dev_dense <T> * A, const h_index * index_a, const dev_h_element <T> * B, const h_index * index_b, dev_temp * tmp_mngr)
  {
    const dev_hierarchical <T> *h_b = B -> getElementHierarchical();
    const dev_low_rank <T> *lr_b = B -> getElementLowRank();
    const dev_dense <T> *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, d_b, index_b, tmp_mngr); }
    if (lr_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, lr_b, index_b, tmp_mngr); }
    if (h_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, h_b, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ static h_ops_tree * generateOps_GEMM (const h_index * self, const dev_low_rank <T> * A, const h_index * index_a, const dev_h_element <T> * B, const h_index * index_b, dev_temp * tmp_mngr)
  {
    const dev_hierarchical <T> *h_b = B -> getElementHierarchical();
    const dev_low_rank <T> *lr_b = B -> getElementLowRank();
    const dev_dense <T> *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, d_b, index_b, tmp_mngr); }
    if (lr_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, lr_b, index_b, tmp_mngr); }
    if (h_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, h_b, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ static h_ops_tree * generateOps_GEMM (const h_index * self, const dev_hierarchical <T> * A, const h_index * index_a, const dev_h_element <T> * B, const h_index * index_b, dev_temp * tmp_mngr)
  {
    const dev_hierarchical <T> *h_b = B -> getElementHierarchical();
    const dev_low_rank <T> *lr_b = B -> getElementLowRank();
    const dev_dense <T> *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, d_b, index_b, tmp_mngr); }
    if (lr_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, lr_b, index_b, tmp_mngr); }
    if (h_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, h_b, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ static h_ops_tree * generateOps_GEMM (const h_index * self, const dev_h_element <T> * A, const h_index * index_a, const dev_h_element <T> * B, const h_index * index_b, dev_temp * tmp_mngr)
  {
    const dev_hierarchical <T> *h_b = B -> getElementHierarchical();
    const dev_low_rank <T> *lr_b = B -> getElementLowRank();
    const dev_dense <T> *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, d_b, index_b, tmp_mngr); }
    if (lr_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, lr_b, index_b, tmp_mngr); }
    if (h_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, h_b, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ cudaError_t loadBinary_ReverseEndian (FILE * stream)
  {
    int real_l = (int) sizeof(T);
    unsigned char * buf = new unsigned char[nx * ny * real_l];

    if (stream != nullptr && fread(buf, sizeof(T), nx * ny, stream) > 0)
    {
#pragma omp parallel if (omp_in_parallel == 0)
      for (int i = 0; i < ny * nx; i++)
      {
        int buf_start = i * real_l;
        for (int bit = 0; bit < real_l / 2; bit++)
        {
          int i1 = buf_start + bit, i2 = buf_start + real_l - bit - 1;
          unsigned char c = buf[i1]; buf[i1] = buf[i2]; buf[i2] = c;
        }
      }

      cudaError_t error = cudaMemcpy (elements, buf, nx * ny * real_l, cudaMemcpyDefault);
      delete[] buf;
      return error;
    }
    else
    {
      printf("Error Reading from File.\n");
      delete[] buf;
      return cudaErrorFileNotFound;
    }

  }

   
  __host__ void print (const int y_start = 0, const int ny_in = 0, const int x_start = 0, const int nx_in = 0) const
  {
    printf("-- %d x %d | ld: %d | addr: %p --\n", ny, nx, ld, elements);
    const int y_end_in = y_start + ny_in, x_end_in = x_start + nx_in;
    const int y_end = (y_end_in > ny || y_end_in <= y_start) ? ny : y_end_in, x_end = (x_end_in > nx || x_end_in <= x_start) ? nx : x_end_in;

    for (int y = y_start > 0 ? y_start : 0; y < y_end; y++)
    {
      for (int x = x_start > 0 ? x_start : 0; x < x_end; x++)
      {
        T e = elements[y * ld + x];
        printf("%.6e ", e);
      }
      printf("\n");
    }

    if (pivoted)
    {
      printf("\n-- Pivot: --\n");
      for (int y = y_start > 0 ? y_start : 0; y < y_end; y++)
      {
        printf("%d ", pivot[y]);
      }
      printf("\n");
    }
    
    printf("\n");
  }

  __host__ void loadTestMatrix (const int x_start = 0, const int y_start = 0)
  {
    for(int i = 0; i < ny; i++)
    {
      const int y = y_start + i;
      for(int j = 0; j < nx; j++)
      {
        const int x = x_start + j;
        elements[i * ld + j] = (T) (1.0 / (1.0 + ((x > y) ? x - y : y - x)));
      }
    }
  }

  __host__ void loadIdentityMatrix ()
  {
    for (int x = 0; x < nx; x++) for (int y = 0; y < ny; y++)
    { elements[y * ld + x] = (T) ((x == y) ? 1 : 0); }
  }

  __host__ void loadRandomMatrix (const double min, const double max, const int seed = 0)
  {
    if (seed > 0) 
    { srand(seed); }
    for(int x = 0; x < nx; x++) for(int y = 0; y < ny; y++)
    { elements[y * ld + x] = (T) (min + ((T) rand() / RAND_MAX) * (max - min)); }
  }

  __host__ void loadRandomOrthMatrix (const int seed = 0)
  {
    if (seed > 0) 
    { srand(seed); }
    for (int x = 0; x < nx; x++) for (int y = 0; y < ny; y++)
    { elements[y * ld + x] = (T) ((x == y) ? 1 : 0); }

    for (int x = 0; x < nx; x++) for (int z = 0; z < x; z++)
    {
      double t = (double) rand() / RAND_MAX, cos = 1. / sqrt(1. + t * t), sin = cos * t;
      for (int y = 0; y < ny; y++)
      {
        const T e1_T = elements[y * ld + x], e2_T = elements[y * ld + z];
        elements[y * ld + x] = cos * e1_T - sin * e2_T;
        elements[y * ld + z] = sin * e1_T + cos * e2_T;
      }
    }
  }

  __host__ dev_dense <T> * matrixMultiplication (const dev_dense <T> *B) const
  {
    dev_dense <T> *C = new dev_dense <T> (B -> nx, ny);
    for(int m = 0; m < ny; m++)
    {
      for(int n = 0; n < B -> nx; n++)
      {
        for(int k = 0; k < nx; k++)
        {
          (C -> elements)[m * (C -> ld) + n] += elements[m * ld + k] * (B -> elements)[k * (B -> ld) + n];
        }
      }
    }
    return C;
  }

  __host__ dev_dense <T> * transpose() const
  {
    dev_dense <T> *C = new dev_dense <T> (ny, nx);
    for (int m = 0; m < ny; m++)
    {
      for (int n = 0; n < nx; n++)
      {
        (C -> elements)[n * (C -> ld) + m] = elements[m * ld + n];
      }
    }
    return C;
  }

  __host__ dev_dense <T> * restoreLU () const
  {
    dev_dense <T> *L = new dev_dense <T>(ny, ny);
    for (int i = 0; i < ny; i++)
    {
      for (int j = 0; j < ny; j++)
      {
        if (i > j && j < nx)
        {
          (L -> elements)[i * ny + j] = elements[i * ld + j];
        }
        else if (i == j)
        {
          (L -> elements)[i * ny + j] = 1;
        }
      }
    }

    dev_dense <T> *U = new dev_dense <T>(nx, ny);
    for (int i = 0; i < ny; i++)
    {
      for (int j = 0; j < nx; j++)
      {
        if (i <= j)
        {
          (U -> elements)[i * nx + j] = elements[i * ld + j];
        }
      }
    }
    
    dev_dense <T> *LU = L -> matrixMultiplication(U);
    delete L;
    delete U;
    return LU;
  }

  __host__ double sqrSum() const
  {
    double sum = 0.0;
    for (int x = 0; x < nx; x++)
    {
      for (int y = 0; y < ny; y++)
      {
        double t = (double)elements[y * ld + x];
        sum += t * t;
      }
    }
    return sum;
  }

  __host__ double L2Error (const dev_dense <T> *matrix) const
  {
    double norm = 0.0; int error_count = 0;
    for(int x = 0; x < nx; x++)
    {
      for(int y = 0; y < ny; y++)
      {
        double t = (double) (elements[y * ld + x] - (matrix -> elements)[y * (matrix -> ld) + x]);
        if (fabs(t) > 1.e-8)
        {
          if (error_count < 10)
          { printf("Error Location: (%d, %d). M1: %.6e M2: %.6e\n", y, x, elements[y * ld + x], (matrix->elements)[y * (matrix->ld) + x]); }
          error_count ++;
        }
        norm += t * t;
      }
    }

    if (error_count > 0)
    { printf("Total Error Locations: %d.\n", error_count); }
    return sqrt(norm / sqrSum());
  }

};


#endif