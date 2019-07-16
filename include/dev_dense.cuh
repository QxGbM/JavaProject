
#pragma once
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

  int shadow_rank;
  T * shadow_u;
  T * shadow_vt;

public:

  __host__ dev_dense (const int nx_in = 0, const int ny_in = 0, const int ld_in = 0, const int shadow_rank_in = 0, const int device_id_in = 0, const bool alloc_pivot = false)
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

      if (shadow_rank_in > 0 && 
        cudaMallocManaged(&shadow_u, ny * shadow_rank_in * sizeof(T), cudaMemAttachGlobal) == cudaSuccess &&
        cudaMallocManaged(&shadow_vt, nx * shadow_rank_in * sizeof(T), cudaMemAttachGlobal) == cudaSuccess)
      { 
        shadow_rank = shadow_rank_in; 
        cudaMemset(shadow_u, 0, ny * shadow_rank_in * sizeof(T));
        cudaMemset(shadow_vt, 0, nx * shadow_rank_in * sizeof(T));
      }
      else
      { shadow_rank = 0; shadow_u = nullptr; shadow_vt = nullptr; }
    }
    else
    { 
      device_id = -1;

      elements = nullptr;
      pivoted = false;
      pivot = nullptr;

      shadow_rank = 0; 
      shadow_u = nullptr; 
      shadow_vt = nullptr;
    }
  }

  __host__ ~dev_dense ()
  {
    cudaFree (elements);
    if (pivoted) 
    { cudaFree (pivot); }
    if (shadow_rank > 0) 
    { cudaFree (shadow_u); cudaFree (shadow_vt); }
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

  __host__ inline int getShadowRank () const
  { return shadow_rank; }

  __host__ inline T * getShadow_U (const int offset = 0) const
  { return &shadow_u[offset]; }

  __host__ inline T * getShadow_VT (const int offset = 0) const
  { return &shadow_vt[offset]; }

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

  __host__ cudaError_t loadBinary (FILE * stream, bool reverse_bytes = true)
  {
    const int buf_size = 8192 * 1024, real_l = (int) sizeof(T);
    const int lines = (buf_size / ld) > ny ? ny : buf_size / ld;
    const int iters = ny / lines, last_lines = ny - iters * lines;

    unsigned char * buf = new unsigned char [buf_size * real_l];

    for (int i0 = 0; i0 < iters; i0++)
    {
      T * elements_row = &elements[i0 * lines * ld];

      if (fread(buf, sizeof(T), nx * lines, stream) > 0)
      {

        if (reverse_bytes)
        #pragma omp parallel for
        for (int i = 0; i < nx * lines; i++)
        {
          int buf_start = i * real_l, i1 = buf_start, i2 = buf_start + real_l - 1;
          for (int bit = 0; bit < real_l / 2; bit++)
          {
            unsigned char c = buf[i1]; buf[i1] = buf[i2]; buf[i2] = c;
            i1 ++; i2 --;
          }
        }

        if (ld > nx)
        {
          for (int i = lines - 1; i >= 0; i--)
          { memmove(&buf[i * ld], &buf[i * nx], nx * real_l); }

          for (int i = 0; i < lines; i++)
          { memset(&buf[i * nx], 0, (ld - nx) * real_l); }
        }

        cudaMemcpy (elements_row, buf, lines * ld * real_l, cudaMemcpyDefault);
      }

    }

    if (last_lines > 0)
    {
      T * elements_row = &elements[iters * lines * ld];

      if (fread(buf, sizeof(T), nx * last_lines, stream) > 0)
      {

        if (reverse_bytes)
        #pragma omp parallel for
        for (int i = 0; i < nx * last_lines; i++)
        {
          int buf_start = i * real_l, i1 = buf_start, i2 = buf_start + real_l - 1;
          for (int bit = 0; bit < real_l / 2; bit++)
          {
            unsigned char c = buf[i1]; buf[i1] = buf[i2]; buf[i2] = c;
            i1 ++; i2 --;
          }
        }

        if (ld > nx)
        {
          for (int i = last_lines - 1; i >= 0; i--)
          { memmove(&buf[i * ld], &buf[i * nx], nx * real_l); }

          for (int i = 0; i < last_lines; i++)
          { memset(&buf[i * nx], 0, (ld - nx) * real_l); }
        }

        cudaMemcpy (elements_row, buf, last_lines * ld * real_l, cudaMemcpyDefault);
      }
    }

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    { fprintf(stderr, "Memcpy: %s\n", cudaGetErrorString(error)); }

    delete[] buf;
    return error;
  }

  __host__ static dev_dense <T> * readStructureFromFile (FILE * stream)
  {
    element_t type;
    void * d = dev_h_element <T> :: readStructureFromFile(stream, &type);

    if (type == hierarchical)
    { return (dev_dense <T> *) d; }
    else
    {
      printf("The Matrix Loaded is not a dense matrix.\n");

      if (type == hierarchical)
      { dev_hierarchical<T> * h = (dev_hierarchical<T> *) d; delete h; }
      else if (type == low_rank)
      { dev_low_rank<T> * lr = (dev_low_rank<T> *) d; delete lr; }

      return nullptr; 
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