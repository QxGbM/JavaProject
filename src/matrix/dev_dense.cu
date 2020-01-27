
#include <definitions.cuh>
#include <matrix/dev_dense.cuh>
#include <matrix/dev_low_rank.cuh>
#include <matrix/dev_hierarchical.cuh>
#include <matrix/dev_hierarchical_element.cuh>
#include <h_ops/dev_hierarchical_index.cuh>
#include <h_ops/dev_hierarchical_ops.cuh>
#include <h_ops/dev_hierarchical_ops_tree.cuh>
#include <dev_temp.cuh>


dev_dense::dev_dense (const int nx_in, const int ny_in, const int ld_in, const int shadow_rank_in, const int device_id_in, const bool alloc_pivot)
{
  nx = nx_in;
  ny = ny_in;
  ld = (nx > ld_in) ? nx : ld_in;

  if (device_id_in >= 0 && cudaSetDevice(device_id_in) == cudaSuccess)
  { 
    device_id = device_id_in;

    if (cudaMallocManaged(&elements, (size_t) ld * ny * real_bits, cudaMemAttachGlobal) == cudaSuccess)
    { cudaMemset(elements, 0, (size_t) ld * ny * real_bits); }
    else
    { elements = nullptr; }
    
    if (alloc_pivot && cudaMallocManaged(&pivot, ny * sizeof(int), cudaMemAttachGlobal) == cudaSuccess)
    { cudaMemset(pivot, 0, ny * sizeof(int)); pivoted = true; }
    else
    { pivot = nullptr; pivoted = false; }

    if (shadow_rank_in > 0 && 
      cudaMallocManaged(&shadow_u, (size_t) ny * shadow_rank_in * real_bits, cudaMemAttachGlobal) == cudaSuccess &&
      cudaMallocManaged(&shadow_vt, (size_t) nx * shadow_rank_in * real_bits, cudaMemAttachGlobal) == cudaSuccess)
    { 
      shadow_rank = shadow_rank_in;
      cudaMemset(shadow_u, 0, (size_t) ny * shadow_rank_in * real_bits);
      cudaMemset(shadow_vt, 0, (size_t) nx * shadow_rank_in * real_bits);
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

dev_dense::~dev_dense ()
{
  cudaFree (elements);
  if (pivoted) 
  { cudaFree (pivot); }
  if (shadow_rank > 0) 
  { cudaFree (shadow_u); cudaFree (shadow_vt); }
}

int dev_dense::getNx () const
{ return nx; }

int dev_dense::getNy () const
{ return ny; }

int dev_dense::getLd () const
{ return ld; }

real_t * dev_dense::getElements (const int offset) const
{ return &elements[offset]; }

int * dev_dense::getPivot (const int offset) const
{ return pivoted ? &pivot[offset / ld] : nullptr; }

int dev_dense::getShadowRank () const
{ return shadow_rank; }

real_t * dev_dense::getShadow_U (const int offset) const
{ return &shadow_u[offset]; }

real_t * dev_dense::getShadow_VT (const int offset) const
{ return &shadow_vt[offset]; }

cudaError_t dev_dense::resize (const int ld_in, const int ny_in)
{
  cudaError_t error = resizeColumn(ld_in);
  return error == cudaSuccess ? resizeRow(ny_in) : error;
}

cudaError_t dev_dense::resizeColumn (const int ld_in)
{
  if (ld_in > 0 && ld_in != ld)
  {
    real_t * e = nullptr;
    cudaError_t error = cudaMallocManaged (&e, (size_t) ld_in * ny * real_bits, cudaMemAttachGlobal);
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

cudaError_t dev_dense::resizeRow (const int ny_in)
{
  if (ny_in > 0 && ny_in != ny)
  {
    real_t * e = nullptr;
    cudaError_t error = cudaMallocManaged (&e, (size_t) ld * ny_in * real_bits, cudaMemAttachGlobal);
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

cudaError_t dev_dense::resizeShadow (const int shadow_rank_in)
{
  if (shadow_rank_in > 0 && shadow_rank_in != shadow_rank)
  {
    if (shadow_rank > 0)
    {
      cudaFree (shadow_u);
      cudaFree (shadow_vt);
    }

    if (shadow_rank_in > 0 && 
      cudaMallocManaged(&shadow_u, (size_t) ny * shadow_rank_in * real_bits, cudaMemAttachGlobal) == cudaSuccess &&
      cudaMallocManaged(&shadow_vt, (size_t) nx * shadow_rank_in * real_bits, cudaMemAttachGlobal) == cudaSuccess)
    { 
      shadow_rank = shadow_rank_in;
      cudaMemset(shadow_u, 0, (size_t) ny * shadow_rank_in * real_bits);
      cudaMemset(shadow_vt, 0, (size_t) nx * shadow_rank_in * real_bits);
    }
    else
    { shadow_rank = 0; shadow_u = nullptr; shadow_vt = nullptr; }
  }
  else if (shadow_rank_in <= 0 && shadow_rank > 0)
  {
    shadow_rank = 0;
    cudaFree (shadow_u);
    cudaFree (shadow_vt);
    shadow_u = nullptr;
    shadow_vt = nullptr;
  }

  return cudaGetLastError();
}

h_ops_tree * dev_dense::generateOps_GETRF (const h_index * self, dev_temp * tmp_mngr)
{
  h_ops_tree * op = new h_ops_tree (getrf, self);

  if (self -> getRank() > 0)
  { 
    op -> resizeChildren(2);
    h_index index_s = h_index (self); index_s.setShadow (self);
    h_ops_tree * op_accm = new h_ops_tree (accum, self, &index_s);
    op -> setChild(op_accm, 0);

    h_ops_tree * op_act = new h_ops_tree (getrf, self);
    op -> setChild(op_act, 1);

    delete op_accm;
    delete op_act;
  }

  return op;
}

h_ops_tree * dev_dense::generateOps_TRSML (const h_index * self, const dev_dense * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  h_ops_tree * op = new h_ops_tree (trsml, index_b, self);

  if (index_b -> getRank() > 0)
  { 
    op -> resizeChildren(2);
    h_index index_s = h_index (index_b); index_s.setShadow (index_b);
    h_ops_tree * op_accm = new h_ops_tree (accum, index_b, &index_s);
    op -> setChild(op_accm, 0);

    h_ops_tree * op_act = new h_ops_tree (trsml, index_b, self);
    op -> setChild(op_act, 1);

    delete op_accm;
    delete op_act;
  }

  return op;
}

h_ops_tree * dev_dense::generateOps_TRSML (const h_index * self, const dev_low_rank * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  h_index index_lr = h_index (index_b); index_lr.setU();
  h_ops_tree * op = new h_ops_tree (trsml, &index_lr, self);
  return op;
}

h_ops_tree * dev_dense::generateOps_TRSML (const h_index * self, const dev_hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  printf("Not implemented.\n");
  return nullptr;
}

h_ops_tree * dev_dense::generateOps_TRSML (const h_index * self, const dev_h_element * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  const dev_hierarchical *h_b = B -> getElementHierarchical();
  const dev_low_rank *lr_b = B -> getElementLowRank();
  const dev_dense *d_b = B -> getElementDense();

  if (d_b != nullptr)
  { return generateOps_TRSML (self, d_b, index_b, tmp_mngr); }
  if (lr_b != nullptr)
  { return generateOps_TRSML (self, lr_b, index_b, tmp_mngr); }
  if (h_b != nullptr)
  { return generateOps_TRSML (self, h_b, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * dev_dense::generateOps_TRSMR (const h_index * self, const dev_dense * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  h_ops_tree * op = new h_ops_tree (trsmr, index_b, self);

  if (index_b -> getRank() > 0)
  { 
    op -> resizeChildren(2);
    h_index index_s = h_index (index_b); index_s.setShadow (index_b);
    h_ops_tree * op_accm = new h_ops_tree (accum, index_b, &index_s);
    op -> setChild(op_accm, 0);

    h_ops_tree * op_act = new h_ops_tree (trsmr, index_b, self);
    op -> setChild(op_act, 1);

    delete op_accm;
    delete op_act;
  }

  return op;
}

h_ops_tree * dev_dense::generateOps_TRSMR (const h_index * self, const dev_low_rank * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  h_index index_lr = h_index (index_b); index_lr.setVT();
  h_ops_tree * op = new h_ops_tree (trsmr, &index_lr, self);
  return op;
}

h_ops_tree * dev_dense::generateOps_TRSMR (const h_index * self, const dev_hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  printf("Not implemented.\n");
  return nullptr;
}

h_ops_tree * dev_dense::generateOps_TRSMR (const h_index * self, const dev_h_element * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  const dev_hierarchical *h_b = B -> getElementHierarchical();
  const dev_low_rank *lr_b = B -> getElementLowRank();
  const dev_dense *d_b = B -> getElementDense();

  if (d_b != nullptr)
  { return generateOps_TRSMR (self, d_b, index_b, tmp_mngr); }
  if (lr_b != nullptr)
  { return generateOps_TRSMR (self, lr_b, index_b, tmp_mngr); }
  if (h_b != nullptr)
  { return generateOps_TRSMR (self, h_b, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * dev_dense::generateOps_ACCM (const h_index * self, const h_index * index_tmp_lr)
{
  if (self -> getRank() > 0)
  {
    h_index index_s = h_index (self); index_s.setShadow (self);
    return new h_ops_tree (accum, &index_s, index_tmp_lr);
  }
  else
  { return new h_ops_tree (accum, self, index_tmp_lr); }
}

h_ops_tree * dev_dense::generateOps_GEMM (const h_index * self, const dev_dense * A, const h_index * index_a, const dev_dense * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  return new h_ops_tree (gemm, self, index_a, index_b);
}

h_ops_tree * dev_dense::generateOps_GEMM (const h_index * self, const dev_low_rank * A, const h_index * index_a, const dev_dense * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  if (self -> getRank() > 0)
  {
    h_index index_s = h_index (self); index_s.setShadow (self);
    return dev_low_rank :: generateOps_GEMM (&index_s, A, index_a, B, index_b, tmp_mngr);
  }
  else
  { return new h_ops_tree (gemm, self, index_a, index_b); }
}

h_ops_tree * dev_dense::generateOps_GEMM (const h_index * self, const dev_hierarchical * A, const h_index * index_a, const dev_dense * B, const h_index * index_b, dev_temp * tmp_mngr)
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

h_ops_tree * dev_dense::generateOps_GEMM (const h_index * self, const dev_h_element * A, const h_index * index_a, const dev_dense * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  const dev_hierarchical *h_a = A -> getElementHierarchical();
  const dev_low_rank *lr_a = A -> getElementLowRank();
  const dev_dense *d_a = A -> getElementDense();

  if (d_a != nullptr)
  { return generateOps_GEMM (self, d_a, index_a, B, index_b, tmp_mngr); }
  if (lr_a != nullptr)
  { return generateOps_GEMM (self, lr_a, index_a, B, index_b, tmp_mngr); }
  if (h_a != nullptr)
  { return generateOps_GEMM (self, h_a, index_a, B, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * dev_dense::generateOps_GEMM (const h_index * self, const dev_dense * A, const h_index * index_a, const dev_low_rank * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  if (self -> getRank() > 0)
  {
    h_index index_s = h_index (self); index_s.setShadow (self);
    return dev_low_rank :: generateOps_GEMM (&index_s, A, index_a, B, index_b, tmp_mngr);
  }
  else
  { return new h_ops_tree (gemm, self, index_a, index_b); }
}

h_ops_tree * dev_dense::generateOps_GEMM (const h_index * self, const dev_low_rank * A, const h_index * index_a, const dev_low_rank * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  if (self -> getRank() > 0)
  {
    h_index index_s = h_index (self); index_s.setShadow (self);
    return dev_low_rank :: generateOps_GEMM (&index_s, A, index_a, B, index_b, tmp_mngr);
  }
  else
  { return new h_ops_tree (gemm, self, index_a, index_b); }
}

h_ops_tree * dev_dense::generateOps_GEMM (const h_index * self, const dev_hierarchical * A, const h_index * index_a, const dev_low_rank * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  if (self -> getRank() > 0)
  {
    h_index index_s = h_index (self); index_s.setShadow (self);
    return dev_low_rank :: generateOps_GEMM (&index_s, A, index_a, B, index_b, tmp_mngr);
  }
  else
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
}

h_ops_tree * dev_dense::generateOps_GEMM (const h_index * self, const dev_h_element * A, const h_index * index_a, const dev_low_rank * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  const dev_hierarchical *h_a = A -> getElementHierarchical();
  const dev_low_rank *lr_a = A -> getElementLowRank();
  const dev_dense *d_a = A -> getElementDense();

  if (d_a != nullptr)
  { return generateOps_GEMM (self, d_a, index_a, B, index_b, tmp_mngr); }
  if (lr_a != nullptr)
  { return generateOps_GEMM (self, lr_a, index_a, B, index_b, tmp_mngr); }
  if (h_a != nullptr)
  { return generateOps_GEMM (self, h_a, index_a, B, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * dev_dense::generateOps_GEMM (const h_index * self, const dev_dense * A, const h_index * index_a, const dev_hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr)
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

h_ops_tree * dev_dense::generateOps_GEMM (const h_index * self, const dev_low_rank * A, const h_index * index_a, const dev_hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  if (self -> getRank() > 0)
  {
    h_index index_s = h_index (self); index_s.setShadow (self);
    return dev_low_rank :: generateOps_GEMM (&index_s, A, index_a, B, index_b, tmp_mngr);
  }
  else
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
}

h_ops_tree * dev_dense::generateOps_GEMM (const h_index * self, const dev_hierarchical * A, const h_index * index_a, const dev_hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr)
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

h_ops_tree * dev_dense::generateOps_GEMM (const h_index * self, const dev_h_element * A, const h_index * index_a, const dev_hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  const dev_hierarchical *h_a = A -> getElementHierarchical();
  const dev_low_rank *lr_a = A -> getElementLowRank();
  const dev_dense *d_a = A -> getElementDense();

  if (d_a != nullptr)
  { return generateOps_GEMM (self, d_a, index_a, B, index_b, tmp_mngr); }
  if (lr_a != nullptr)
  { return generateOps_GEMM (self, lr_a, index_a, B, index_b, tmp_mngr); }
  if (h_a != nullptr)
  { return generateOps_GEMM (self, h_a, index_a, B, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * dev_dense::generateOps_GEMM (const h_index * self, const dev_dense * A, const h_index * index_a, const dev_h_element * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  const dev_hierarchical *h_b = B -> getElementHierarchical();
  const dev_low_rank *lr_b = B -> getElementLowRank();
  const dev_dense *d_b = B -> getElementDense();

  if (d_b != nullptr)
  { return generateOps_GEMM (self, A, index_a, d_b, index_b, tmp_mngr); }
  if (lr_b != nullptr)
  { return generateOps_GEMM (self, A, index_a, lr_b, index_b, tmp_mngr); }
  if (h_b != nullptr)
  { return generateOps_GEMM (self, A, index_a, h_b, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * dev_dense::generateOps_GEMM (const h_index * self, const dev_low_rank * A, const h_index * index_a, const dev_h_element * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  const dev_hierarchical *h_b = B -> getElementHierarchical();
  const dev_low_rank *lr_b = B -> getElementLowRank();
  const dev_dense *d_b = B -> getElementDense();

  if (d_b != nullptr)
  { return generateOps_GEMM (self, A, index_a, d_b, index_b, tmp_mngr); }
  if (lr_b != nullptr)
  { return generateOps_GEMM (self, A, index_a, lr_b, index_b, tmp_mngr); }
  if (h_b != nullptr)
  { return generateOps_GEMM (self, A, index_a, h_b, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * dev_dense::generateOps_GEMM (const h_index * self, const dev_hierarchical * A, const h_index * index_a, const dev_h_element * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  const dev_hierarchical *h_b = B -> getElementHierarchical();
  const dev_low_rank *lr_b = B -> getElementLowRank();
  const dev_dense *d_b = B -> getElementDense();

  if (d_b != nullptr)
  { return generateOps_GEMM (self, A, index_a, d_b, index_b, tmp_mngr); }
  if (lr_b != nullptr)
  { return generateOps_GEMM (self, A, index_a, lr_b, index_b, tmp_mngr); }
  if (h_b != nullptr)
  { return generateOps_GEMM (self, A, index_a, h_b, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * dev_dense::generateOps_GEMM (const h_index * self, const dev_h_element * A, const h_index * index_a, const dev_h_element * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  const dev_hierarchical *h_b = B -> getElementHierarchical();
  const dev_low_rank *lr_b = B -> getElementLowRank();
  const dev_dense *d_b = B -> getElementDense();

  if (d_b != nullptr)
  { return generateOps_GEMM (self, A, index_a, d_b, index_b, tmp_mngr); }
  if (lr_b != nullptr)
  { return generateOps_GEMM (self, A, index_a, lr_b, index_b, tmp_mngr); }
  if (h_b != nullptr)
  { return generateOps_GEMM (self, A, index_a, h_b, index_b, tmp_mngr); }

  return nullptr;
}

cudaError_t dev_dense::loadBinary (FILE * stream, const bool reverse_bytes)
{
  const int buf_size = 8192 * 1024, real_l = (int) real_bits;
  const int lines = (buf_size / ld) > ny ? ny : buf_size / ld;
  const int iters = ny / lines, last_lines = ny - iters * lines;

  unsigned char * buf = new unsigned char [buf_size * real_l];

  for (int i0 = 0; i0 < iters; i0++)
  {
    real_t * elements_row = &elements[i0 * lines * ld];

    if (fread(buf, real_bits, (size_t) nx * lines, stream) > 0)
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
        { memmove(&buf[i * ld], &buf[i * nx], (size_t) real_l * nx); }

        for (int i = 0; i < lines; i++)
        { memset(&buf[i * nx], 0, (size_t) real_l * ((size_t) ld - nx)); }
      }

      cudaMemcpy (elements_row, buf, (size_t) lines * ld * real_l, cudaMemcpyDefault);
    }

  }

  if (last_lines > 0)
  {
    real_t * elements_row = &elements[iters * lines * ld];

    if (fread(buf, real_bits, (size_t) nx * last_lines, stream) > 0)
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
        { memmove(&buf[i * ld], &buf[i * nx], (size_t) real_l * nx); }

        for (int i = 0; i < last_lines; i++)
        { memset(&buf[i * nx], 0, (size_t) real_l * ((size_t) ld - nx)); }
      }

      cudaMemcpy (elements_row, buf, (size_t) last_lines * ld * real_l, cudaMemcpyDefault);
    }
  }

  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess)
  { fprintf(stderr, "Memcpy: %s\n", cudaGetErrorString(error)); }

  delete[] buf;
  return error;
}

dev_dense * dev_dense::readStructureFromFile (FILE * stream, const int shadow_rank)
{
  element_t type;
  void * d = dev_h_element :: readStructureFromFile(stream, &type, shadow_rank);

  if (type == dense)
  { return (dev_dense *) d; }
  else
  {
    printf("The Matrix Loaded is not a dense matrix.\n");

    if (type == hierarchical)
    { dev_hierarchical * h = (dev_hierarchical *) d; delete h; }
    else if (type == low_rank)
    { dev_low_rank * lr = (dev_low_rank *) d; delete lr; }

    return nullptr; 
  }

}

dev_dense * dev_dense::readFromFile (const char * file_name, const int shadow_rank, const bool reverse_bytes)
{
  char str[32], bin[32];
  strcpy(str, file_name); strcat(str, ".struct");
  strcpy(bin, file_name); strcat(bin, ".bin");

  FILE * stream = fopen(str, "r");
  dev_dense * a = dev_dense :: readStructureFromFile (stream, shadow_rank);
  fclose(stream);

  if (a != nullptr)
  {
    stream = fopen(bin, "rb");
    a -> loadBinary(stream, reverse_bytes);
    fclose(stream);
  }

  return a;
}
   
void dev_dense::print (const int y_start, const int ny_in, const int x_start, const int nx_in) const
{
  printf("-- %d x %d | ld: %d | addr: %p --\n", ny, nx, ld, elements);
  const int y_end_in = y_start + ny_in, x_end_in = x_start + nx_in;
  const int y_end = (y_end_in > ny || y_end_in <= y_start) ? ny : y_end_in, x_end = (x_end_in > nx || x_end_in <= x_start) ? nx : x_end_in;

  for (int y = y_start > 0 ? y_start : 0; y < y_end; y++)
  {
    for (int x = x_start > 0 ? x_start : 0; x < x_end; x++)
    {
      real_t e = elements[y * ld + x];
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


real_t dev_dense::sqrSum() const
{
  real_t sum = 0.0;
  for (int x = 0; x < nx; x++)
  {
    for (int y = 0; y < ny; y++)
    {
      real_t t = (real_t) elements[y * ld + x];
      sum += t * t;
    }
  }
  return sum;
}

real_t dev_dense::L2Error (const dev_dense * matrix) const
{
  real_t norm = 0.0; int error_count = 0;
  for(int x = 0; x < nx; x++)
  {
    for(int y = 0; y < ny; y++)
    {
      real_t t = (real_t) (elements[y * ld + x] - (matrix -> elements)[y * (matrix -> ld) + x]);
      if (fabs(t) > 1.e-8)
      {
        if (error_count < 10)
        { printf("Error Location: (%d, %d). M1: %.6e M2: %.6e\n", y, x, elements[y * ld + x], (matrix -> elements)[y * (matrix -> ld) + x]); }
        error_count ++;
      }
      norm += t * t;
    }
  }

  if (error_count > 0)
  { printf("Total Error Locations: %d.\n", error_count); }
  return sqrt(norm / sqrSum());
}



