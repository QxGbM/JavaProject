#ifndef _DEV_LOW_RANK_FUNCS_CUH
#define _DEV_LOW_RANK_FUNCS_CUH

#include <pspl.cuh>

template <class T> __device__ void blockRotateColumns (T * col1, T * col2, const int ny, const int ld, const double sine, const double cosine)
{
  for (int i = thread_rank(); i < ny; i += block_dim())
  {
    const T col1_T = col1[i * ld], col2_T = col2[i * ld];
    col1[i * ld] = cosine * col1_T - sine * col2_T;
    col2[i * ld] = sine * col1_T + cosine * col2_T;
  }
}

template <class T> __device__ T warpAllReduceSum (T value)
{
  for (int mask = warpSize / 2; mask > 0; mask /= 2)
  { value += __shfl_xor_sync(0xffffffff, value, mask, warpSize); }
  return value;
}

template <class T> __device__ T blockAllReduceSum (T value, T * shm)
{
  value = warpAllReduceSum <T> (value);

  if (lane_rank() == 0)
  { shm[warp_rank()] = value; }
  __syncthreads();

  if (block_dim() > warpSize && warp_rank() == 0)
  {
    value = 0;
    for (int i = lane_rank(); i < num_warps(); i += warpSize)
    { value += shm[i]; }

    value = warpAllReduceSum <T> (value);
    if (lane_rank() == 0)
    { shm[0] = value; }
  }
  __syncthreads();

  return shm[0];
}

template <class T> __device__ T blockVectorMultiplication (const T * vec, const int length, const int ld, T * shm)
{
  T thread_sum = 0;
  for (int i = thread_rank(); i < length; i += block_dim())
  { thread_sum += vec[i * ld] * vec[i * ld]; }

  return blockAllReduceSum <T> (thread_sum, shm);
}

template <class T> __device__ T blockVectorMultiplication (const T * vec1, const T * vec2, const int length, const int ld_1, const int ld_2, T * shm)
{
  T thread_sum = 0;
  for (int i = thread_rank(); i < length; i += block_dim())
  { thread_sum += vec1[i * ld_1] * vec2[i * ld_2]; }

  return blockAllReduceSum <T> (thread_sum, shm);
}

template <class T> __device__ double blockJacobiRotate (T * A, T * VT, const int nx, const int ny, const int ld_a, const int ld_v, const int col_i, const int col_j, T * shm)
{
  const T s_ii = blockVectorMultiplication <T> (&A[col_i], ny, ld_a, &shm[0]);
  const T s_jj = blockVectorMultiplication <T> (&A[col_j], ny, ld_a, &shm[num_warps()]);
  const T s_ij = blockVectorMultiplication <T> (&A[col_i], &A[col_j], ny, ld_a, ld_a, &shm[2 * num_warps()]);

  if (s_ii > s_jj)
  {
    blockSwapColumns <T> (&A[col_i], &A[col_j], ny, ld_a);
    blockSwapColumns <T> (&VT[col_i], &VT[col_j], nx, ld_v);
  }

  double * shm_double = (double *) &shm[3 * num_warps()];
  if (thread_rank() == 0)
  {
    const double torque = (double) ((s_jj - s_ii) / (2.0 * s_ij));
    const int sign_torque = (int) (torque >= 0.0) * 2 - 1;
    const double abs_torque = sign_torque * torque;
    const double tangent = sign_torque / (abs_torque + sqrt(1.0 + torque * torque));
    shm_double[2] = 1.0 / (sqrt(1.0 + tangent * tangent));
    shm_double[1] = shm_double[2] * tangent;
    shm_double[0] = (s_ij > 0) ? (double) s_ij : (double) -s_ij;
  }
  __syncthreads();

  blockRotateColumns <T> (&A[col_i], &A[col_j], ny, ld_a, shm_double[1], shm_double[2]);
  blockRotateColumns <T> (&VT[col_i], &VT[col_j], nx, ld_v, shm_double[1], shm_double[2]);
  __syncthreads();

  return shm_double[0];
}

template <class T> __device__ void resetVT (T * VT, const int n, const int ld)
{
  for (int i = thread_rank(); i < n * n; i++)
  {
    const int row = i / n, col = i - row * n;
    VT[row * ld + col] = (int) (row == col);
  }
}

template <class T> __device__ int blockJacobiSVD (T * A, T * VT, const int nx, const int ny, const int ld_a, const int ld_v, const double epi, const int iter_limit, T * shm)
{
  resetVT (VT, nx, ld_v);
  int * iter = (int *) &shm[0], * loop_counter = (int *) &shm[1];
  if (thread_rank() == 0) 
  { *iter = 1; *loop_counter = 0; }
  __syncthreads();

  while (*iter && *loop_counter < iter_limit)
  {
    if (thread_rank() == 0)
    { *iter = 0; (*loop_counter) ++; }

    for (int i = 1; i < nx; i++)
    {
      for (int j = 0; j < i; j++)
      {
        const double conv = blockJacobiRotate <T> (A, VT, nx, ny, ld_a, ld_v, i, j, &shm[2]);
        if (thread_rank() == 0 && conv > epi)
        { *iter = 1; }
      }
    }

    __syncthreads();
  }
  return *loop_counter;
}

template <class T> __device__ double sumSquaredDifference (const T * A, const T * B, const int nx, const int ny, const int ld_a, const int ld_b, T * shm)
{
  double thread_sum = 0.;
  for (int i = thread_rank(); i < nx * ny; i += block_dim())
  { 
    const int row = i / nx, col = i - row * nx;
    const double diff = A[row * ld_a + col] * B[row * ld_b + col];
    thread_sum += diff * diff; 
  }

  return blockAllReduceSum <double> (thread_sum, (double *) shm);
}



#endif