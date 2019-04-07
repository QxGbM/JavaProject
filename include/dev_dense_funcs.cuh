
#ifndef _DEV_DENSE_FUNCS_CUH
#define _DEV_DENSE_FUNCS_CUH

#include <pspl.cuh>

template <class T>
/* A convinient call to copy from shared memory to global or vice versa. */
__device__ void matrixCopy (const T * from, T * to, const int nx, const int ny, const int ld_from, const int ld_to)
{
  for (int i = thread_rank(); i < nx * ny; i += block_dim())
  {
    const int row = i / nx, col = i - row * nx;
    to[row * ld_to + col] = from[row * ld_from + col];
  }
}

template <class T> 
/* Scalar of a vector of length n. */
__device__ void blockVectorScalar(const T scale, T * V, const int n)
{
  for (int i = thread_rank(); i < n; i += block_dim())
  { V[i] = (scale == 0) ? 0 : V[i] * scale; }
}

template <class T> 
/* Scalar of a matrix of ny by nx. */
__device__ void blockDenseScalar (const T scale, T * M, const int nx, const int ny, const int ld)
{
  for (int i = thread_rank(); i < nx * ny; i += block_dim())
  { 
    const int row = i / nx, col = i - row * nx;
    M[row * ld + col] = (scale == 0) ? 0 : M[row * ld + col] * scale;
  }
}

template <class T> 
/* A has dimension m * k, B has dimension k * n, matrix has dimension m * n. matrix = old_matrix - A * B. Scanning row major. */
__device__ void blockDenseGemm_RM (T * M, const T * A, const T * B, const int m, const int n, const int k, const int ld_m, const int ld_a, const int ld_b)
{
  for (int i = thread_rank(); i < m * n; i += block_dim())
  {
    const int row = i / n, col = i - row * n;
    T accum = 0;
    for (int j = 0; j < k; j++)
    { accum += A[row * ld_a + j] * B[j * ld_b + col]; }
    M[row * ld_m + col] -= accum;
  }
}

/* An overloaded version gemm that uses alpha = -1 and beta = 1. Caches columns in L1, scanning row major. */
template <class T> 
__device__ void blockDenseGemm_Cshm_RM (T * M, const T * A, const T * B, const int m, const int n, const int k, const int ld_m, const int ld_a, const int ld_b, T * shm, const int shm_size)
{
  const int step_size = shm_size / k;

#pragma unroll
  for (int col = 0; col < n; col += step_size)
  {
    const int cols_remaining = n - col, num_cols = (cols_remaining > step_size) ? step_size : cols_remaining;
    
    matrixCopy <T> (&B[col], &shm[0], num_cols, k, ld_b, num_cols);
    __syncthreads();

    blockDenseGemm_RM <T> (&M[col], A, &shm[0], m, num_cols, k, ld_m, ld_a, num_cols);
    __syncthreads();
  }
}

template <class T> 
/* An overloaded version gemm that uses alpha = -1, beta = 1 and k = 1, scanning in row major. */
__device__ void blockDenseGemm_K1_RM (T * M, const T * A, const T * B, const int m, const int n, const int ld_m, const int ld_a)
{
  for (int i = thread_rank(); i < m * n; i += block_dim())
  {
    const int row = i / n, col = i - row * n;
    M[row * ld_m + col] -= A[row * ld_a] * B[col];
  }
}

template <class T> 
/* Find the index of the largest absolute value element across the warp. Returns lane number [0, 31]. */
__device__ int warpReduceMax_Index (const T max_in)
{
  T max = max_in; int max_lane = lane_rank();

  for (int mask = warpSize / 2; mask > 0; mask /= 2)
  {
    const T s_max = __shfl_xor_sync (0xffffffff, max, mask, warpSize);
    const int s_lane = __shfl_xor_sync (0xffffffff, max_lane, mask, warpSize);
    if (s_max > max || (s_max == max && s_lane < max_lane))
    { max = s_max; max_lane = s_lane; }
  }

  return max_lane;
}

template <class T> 
/* Find the index of the largest absolute value element in matrix[0], matrix[1], ... matrix[n-1]. Returns [0, n-1]. */
__device__ int blockReduceMax_Index (const T * M, const int n, int * shm)
{
  T max = 0; int index = 0;
  
  for (int i = thread_rank(); i < n; i += block_dim())
  {
    const T value = abs (M[i]);
    if (value > max)
    { max = value; index = i; }
  }

  if (lane_rank() == warpReduceMax_Index(max))
  { shm[warp_rank()] = index; }
  __syncthreads();

  if (num_warps() > 1 && warp_rank() == 0)
  {
    max = 0; index = 0;
    for (int i = lane_rank(); i < num_warps(); i += warpSize)
    {
      const T value = abs (M[ shm[i] ]);
      if (value > max)
      { max = value; index = shm[i]; }
    }

    if (lane_rank() == warpReduceMax_Index(max))
    { shm[0] = index; }
  }

  __syncthreads(); 

  return shm[0];
}

template <class T> 
/* Exchange row1[0] with row2[0], row1[1] with row2[1], ... row1[n-1] with row2[n-1]. */
__device__ void blockSwapRows (T * row1, T * row2, const int n)
{
  for (int i = thread_rank(); i < n; i += block_dim())
  { const T t = row1[i]; row1[i] = row2[i]; row2[i] = t; }
}

template <class T> 
/* Exchange col1[0] with col2[0], col1[1] with col2[1], ... col1[n-1] with col2[n-1]. */
__device__ void blockSwapColumns (T * col1, T * col2, const int n, const int ld)
{
  for (int i = thread_rank(); i < n; i += block_dim())
  { const T t = col1[i * ld]; col1[i * ld] = col2[i * ld]; col2[i * ld] = t; }
}

template <class T> 
/* Using a group of threads to apply pivot the pivot swaps to the matrix. Recover flag retrieves original matrix. Utilizes L1. */
__device__ void blockApplyPivot (T * M, const int * p, const int nx, const int ny, const int ld, const bool recover, T * shm, const int shm_size)
{
  const int step_size = shm_size / ny;

  for (int n = 0; n < nx; n += step_size)
  {
    const int cols = (nx - n > step_size) ? step_size : nx - n;
    for (int i = thread_rank(); i < cols * ny; i++)
    {
      const int row = i / cols, col = i - row * cols, target = p[row];
      if (recover)
      { shm[target * cols + col] = M[row * ld + n + col]; }
      else
      { shm[row * cols + col] = M[target * ld + n + col]; }
    }
    __syncthreads();

    matrixCopy(&shm[0], &M[n], cols, ny, cols, ld);
    __syncthreads();
  }
}

/* Set pivot[0] = 0, pivot[1] = 1, ... pivot[n-1] = n-1. */
__device__ void resetPivot (int *p, const int n)
{
  for (int i = thread_rank(); i < n; i += block_dim())
  { p[i] = i; }
}

template <class T> 
/* Pivoted LU decomposition of matrix of ny by nx, utilizes L1 cache. */
__device__ void blockDenseGetrf_shm (T * M, const int nx, const int ny, const int ld, int *p, T * shm)
{
  if (p != nullptr) { resetPivot(p, ny); }

  for (int i = 0; i < nx && i < ny; i++)
  {
    matrixCopy <T> (&M[i * ld + i], &shm[0], 1, ny - i, ld, 1);
    __syncthreads();

    if (p != nullptr)
    {
      const int target = i + blockReduceMax_Index <T> (&shm[0], ny - i, (int *) &shm[ny - i]);

      if (target != i)
      {
        blockSwapRows(&M[target * ld], &M[i * ld], nx);
        if (thread_rank() == 0)
        { 
          int t0 = p[target]; p[target] = p[i]; p[i] = t0;
          T t1 = shm[0]; shm[0] = shm[target - i]; shm[target - i] = t1;
        }
      }
      __syncthreads();
    }

    blockVectorScalar <T> (1.0 / shm[0], &shm[1], ny - (i + 1));
    __syncthreads();

    blockDenseGemm_K1_RM <T> (&M[(i + 1) * ld + (i + 1)], &shm[1], &M[i * ld + (i + 1)], ny - (i + 1), nx - (i + 1), ld, 1);

    matrixCopy <T> (&shm[0], &M[i * ld + i], 1, ny - i, 1, ld);
    __syncthreads();
  }
}

template <class T>
/* L is ny_l x nx_l lower triangular and unit diagonal, B is ny_l by nx_b, solves L x X = B, overwrites X in B. */
__device__ void blockDenseTrsmL (T * B, const T * L, const int nx_b, const int ny_b, const int nx_l, const int ld_b, const int ld_l)
{
  for (int i = 0; i < nx_l && i + 1 < ny_b; i++)
  { 
    blockDenseGemm_K1_RM <T> (&B[(i + 1) * ld_b], &L[(i + 1) * ld_l + i], &B[i * ld_b], ny_b - (i + 1), nx_b, ld_b, ld_l);
    __syncthreads();
  }
}

template <class T>
/* U is ny_u x nx_u upper triangular and not unit diagonal, B is ny_b by nx_u, solves X x U = B, overwrites X in B. Utilizes L1 cache. */
 __device__ void blockDenseTrsmR_shm (T * B, const T * U, const int nx_b, const int ny_b, const int ny_u, const int ld_b, const int ld_u, T * shm)
{

  for (int i = 0; i < nx_b && i < ny_u; i++)
  {
    matrixCopy <T> (&B[i], &shm[0], 1, ny_b, ld_b, 1);

    blockVectorScalar <T> (1.0 / U[i * ld_u + i], &shm[0], ny_b);
    __syncthreads();

    if (nx_b > i + 1)
    { blockDenseGemm_K1_RM <T> (&B[i + 1], &shm[0], &U[i * ld_u + (i + 1)], ny_b, nx_b - (i + 1), ld_b, 1); }

    matrixCopy <T> (&shm[0], &B[i], 1, ny_b, 1, ld_b);
    __syncthreads();
  }
}


#endif