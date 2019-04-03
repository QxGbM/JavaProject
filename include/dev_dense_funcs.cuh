
#ifndef _DEV_DENSE_FUNCS_CUH
#define _DEV_DENSE_FUNCS_CUH

#include <pspl.cuh>

/* Scalar of a matrix of ny by nx. */
template <class T> __device__ void blockDenseScalar (const T scale, T * M, const int nx, const int ny, const int ld)
{
  for (int i = thread_rank(); i < nx * ny; i += block_dim())
  { 
    const int row = i / nx, col = i - row * nx;
    M[row * ld + col] = (scale == 0) ? 0 : M[row * ld + col] * scale;
  }
  __syncthreads();
}

/* A has dimension m * k, B has dimension k * n, matrix has dimension m * n. matrix = alpha * A * B + beta * old_matrix. */
template <class T> __device__ void blockDenseGemm (const double alpha, const double beta, T * M, const T * A, const T * B, const int m, const int n, const int k, const int ld_m, const int ld_a, const int ld_b)
{
  for (int i = thread_rank(); i < m * n; i += block_dim())
  { 
    const int row = i / n, col = i - row * n;
    const T old = (beta == 0) ? 0 : beta * M[row * ld_m + col];
    T accum = 0;
    if (alpha != 0)
    {
      for (int j = 0; j < k; j++)
      { accum += A[row * ld_a + j] * B[j * ld_b + col]; }
      accum *= alpha;
    }
    M[row * ld_m + col] = old + accum;
  }
  __syncthreads();
}

/* An overloaded version gemm that uses alpha = -1 and beta = 1. */
template <class T> __device__ void blockDenseGemm (T * M, const T * A, const T * B, const int m, const int n, const int k, const int ld_m, const int ld_a, const int ld_b)
{
  for (int i = thread_rank(); i < m * n; i += block_dim())
  {
    const int row = i / n, col = i - row * n;
    T accum = 0;
    for (int j = 0; j < k; j++)
    { accum += A[row * ld_a + j] * B[j * ld_b + col]; }
    M[row * ld_m + col] -= accum;
  }
  __syncthreads();
}

/* Find the index of the largest absolute value element in matrix[0], matrix[ld], ... matrix[(n-1) * ld]. */
template <class T> __device__ int blockAllFindRowPivot(const T * M, const int n, const int ld)
{
  const int warp_id = warp_rank();
  const int lane_id = lane_rank();

  int index = 0;
  T max = 0;

  /* Load all row entries in warps: Each warp can handle more than 1 warpsize of data or no data. */
  for (int i = thread_rank(); i < n; i += block_dim())
  {
    const T value = abs(M[i * ld]);
    if (value > max)
    { max = value; index = i; }
  }

  /* Reduction in all warps. No need to explicitly sync because warp shfl implies synchronization. */
  for (int mask = warpSize / 2; mask > 0; mask /= 2)
  {
    const T s_max = __shfl_xor_sync(0xffffffff, max, mask, warpSize);
    const int s_index = __shfl_xor_sync(0xffffffff, index, mask, warpSize);
    if (s_max > max)
    { max = s_max; index = s_index; }
  }

  __shared__ T shm_max [MAX_WARPS];
  __shared__ int shm_index [MAX_WARPS];

  /* The first lane of each warp writes into their corresponding shared memory slot. */
  if (lane_id == 0) { shm_max[warp_id] = max; shm_index[warp_id] = index; }

  /* Sync here to make sure shared mem is properly initialized, and reductions in all warps are completed. */
  __syncthreads(); 

  /* Do the final reduction in the first warp, if there are more than 1 warp. */
  if (block_dim() > warpSize && warp_id == 0)
  {
    for (int i = lane_id; i < num_warps(); i += warpSize)
    {
      const T value = shm_max[i];
      if (value > max)
      { max = value; index = shm_index[i]; }
    }

    for (int mask = warpSize / 2; mask > 0; mask /= 2)
    {
      const T s_max = __shfl_xor_sync(0xffffffff, max, mask, warpSize);
      const int s_index = __shfl_xor_sync(0xffffffff, index, mask, warpSize);
      /* Uses more strict comparison to resolve ties. */
      if (s_max > max || (s_max == max && s_index < index))
      { max = s_max; index = s_index; }
    }

    if (lane_id == 0)
    {
      shm_max[lane_id] = max;
      shm_index[lane_id] = index;
    }
  }

  /* Sync here to stop other warps and waits for warp 0. */
  __syncthreads(); 

  return shm_index[0];
}

/* Exchange row1[0] with row2[0], row1[1] with row2[1], ... row1[n-1] with row2[n-1]. */
template <class T> __device__ void blockSwapNSeqElements(T *row1, T *row2, const int n)
{
  /* Using a group of threads to exchange all elements in row with target row. */
  const int thread_id = thread_rank();
  const int block_size = block_dim();

  /* swapping n elements in two rows. */
  for (int i = thread_id; i < n; i += block_size) 
  { const T t = row1[i]; row1[i] = row2[i]; row2[i] = t; }
}

/* Using a group of threads to apply pivot the pivot swaps to the matrix. Recover flag retrieves original matrix. */
template <class T> __device__ void blockApplyPivot(T * M, const int * p, const int nx, const int ny, const int ld, const bool recover)
{
  for (int i = 0; i < ny; i++)
  {
    __shared__ bool smallest_row_in_cycle;
    if (thread_rank() == 0)
    {
      smallest_row_in_cycle = true;
      int swapping_with = p[i];

      while (smallest_row_in_cycle && swapping_with != i)
      {
        if (swapping_with < i) { smallest_row_in_cycle = false; }
        swapping_with = p[swapping_with];
      }
    }
    __syncthreads();

    if (smallest_row_in_cycle)
    {
      int source_row = i, swapping_with = p[i];
      while (swapping_with != i)
      {
        blockSwapNSeqElements <T> (&M[source_row * ld], &M[swapping_with * ld], nx);
        source_row = recover ? i : swapping_with;
        swapping_with = p[swapping_with];
      }
    }
    __syncthreads();
  }
}

/* Set pivot[0] = 0, pivot[1] = 1, ... pivot[n-1] = n-1. */
__device__ void resetPivot(int *p, const int n)
{
  const int thread_id = thread_rank();
  const int block_size = block_dim();
  for (int i = thread_id; i < n; i += block_size)
  {
    p[i] = i;
  }
}

/* Pivoted LU decomposition of matrix of ny by nx. */
template <class T> __device__ void blockDenseGetrf (T * M, const int nx, const int ny, const int ld, int *p = nullptr)
{
  if (p != nullptr) { resetPivot(p, ny); }

  for (int i = 0; i < nx && i < ny; i++)
  {
    if (p != nullptr)
    {
      const int target = i + blockAllFindRowPivot <T> (&M[i * ld + i], ny - i, ld);

      if (target != i)
      {
        blockSwapNSeqElements <T> (&M[target * ld], &M[i * ld], nx);
        if (thread_rank() == 0) 
        { int t = p[target]; p[target] = p[i]; p[i] = t; }
        __syncthreads();
      }
    }

    blockDenseScalar <T> (1.0 / M[i * ld + i], &M[(i + 1) * ld + i], 1, ny - (i + 1), ld);

    blockDenseGemm <T> (&M[(i + 1) * ld + (i + 1)], &M[(i + 1) * ld + i], &M[i * ld + (i + 1)], ny - (i + 1), nx - (i + 1), 1, ld, ld, ld);
  }
}

/* L is ny_l x nx_l lower triangular and unit diagonal, B is ny_l by nx_b, solves L x X = B, overwrites X in B. */
template <class T> __device__ void blockDenseTrsmL (T * B, const T * L, const int nx_b, const int ny_b, const int nx_l, const int ld_b, const int ld_l)
{
  for (int i = 0; i < nx_l && i + 1 < ny_b; i++)
  { blockDenseGemm <T> (&B[(i + 1) * ld_b], &L[(i + 1) * ld_l + i], &B[i * ld_b], ny_b - (i + 1), nx_b, 1, ld_b, ld_l, ld_b); }
}

/* U is ny_u x nx_u upper triangular and not unit diagonal, B is ny_b by nx_u, solves X x U = B, overwrites X in B. */
template <class T> __device__ void blockDenseTrsmR (T * B, const T * U, const int nx_b, const int ny_b, const int ny_u, const int ld_b, const int ld_u)
{
  for (int i = 0; i < nx_b && i < ny_u; i++)
  {
    blockDenseScalar <T> (1.0 / U[i * ld_u + i], &B[i], 1, ny_b, ld_b);
    if (nx_b - i > 1)
    { blockDenseGemm <T> (&B[i + 1], &B[i], &U[i * ld_u + (i + 1)], ny_b, nx_b - (i + 1), 1, ld_b, ld_b, ld_u); }
  }
}


#endif