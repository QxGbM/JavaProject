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

template <class T> __device__ T blockAllReduceSum (T value)
{
  value = warpAllReduceSum(value);

  __shared__ T warp_sum[MAX_WARPS];
  if (lane_rank() == 0)
  { warp_sum[warp_rank()] = value; }
  __syncthreads();

  if (block_dim() > warpSize && warp_rank() == 0)
  {
    value = 0;
    for (int i = lane_rank(); i < num_warps(); i += warpSize)
    { value += warp_sum[i]; }

    value = warpAllReduceSum(value);
    if (lane_rank() == 0)
    { warp_sum[0] = value; }
  }
  __syncthreads();

  return warp_sum[0];
}

template <class T> __device__ T blockVectorMultiplication (const T * vec, const int length, const int ld)
{
  T thread_sum = 0;
  for (int i = thread_rank(); i < length; i += block_dim())
  { thread_sum += vec[i * ld] * vec[i * ld]; }

  return blockAllReduceSum(thread_sum);
}

template <class T> __device__ T blockVectorMultiplication (const T * vec1, const T * vec2, const int length, const int ld_1, const int ld_2)
{
  T thread_sum = 0;
  for (int i = thread_rank(); i < length; i += block_dim())
  { thread_sum += vec1[i * ld_1] * vec2[i * ld_2]; }

  return blockAllReduceSum(thread_sum);
}

template <class T> __device__ double blockJacobiRotate (T * A, T * VT, const int nx, const int ny, const int ld_a, const int ld_v, const int col_i, const int col_j)
{
  const T s_ii = blockVectorMultiplication <T> (&A[col_i], ny, ld_a);
  const T s_jj = blockVectorMultiplication <T> (&A[col_j], ny, ld_a);
  const T s_ij = blockVectorMultiplication <T> (&A[col_i], &A[col_j], ny, ld_a, ld_a);

  if (s_ii > s_jj)
  {
    blockSwapColumns <T> (&A[col_i], &A[col_j], ny, ld_a);
    blockSwapColumns <T> (&VT[col_i], &VT[col_j], nx, ld_v);
  }

  __shared__ double sine, cosine, conv;
  if (thread_rank() == 0)
  {
    const double torque = (double) ((s_jj - s_ii) / (2.0 * s_ij));
    const int sign_torque = (int) (torque >= 0.0) * 2 - 1;
    const double abs_torque = sign_torque * torque;
    const double tangent = sign_torque / (abs_torque + sqrt(1.0 + torque * torque));
    cosine = 1.0 / (sqrt(1.0 + tangent * tangent));
    sine = cosine * tangent;
    conv = (s_ij > 0) ? (double) s_ij : (double) -s_ij;
  }
  __syncthreads();

  blockRotateColumns <T> (&A[col_i], &A[col_j], ny, ld_a, sine, cosine);
  blockRotateColumns <T> (&VT[col_i], &VT[col_j], nx, ld_v, sine, cosine);
  __syncthreads();

  return conv;
}

template <class T> __device__ void resetVT (T * VT, const int n, const int ld)
{
  for (int i = thread_rank(); i < n * n; i++)
  {
    const int row = i / n, col = i - row * n;
    VT[row * ld + col] = (int) (row == col);
  }
}

template <class T> __device__ int blockJacobiSVD (T * A, T * VT, const int nx, const int ny, const int ld_a, const int ld_v, const double epi, const int iter_limit)
{
  __shared__ bool iter;
  __shared__ int iter_counter;

  resetVT (VT, nx, ld_v);
  if (thread_rank() == 0) 
  { iter_counter = 0; iter = true; }
  __syncthreads();

  while (iter && iter_counter < iter_limit)
  {
    if (thread_rank() == 0)
    { iter = false; iter_counter ++; }

    for (int i = 1; i < nx; i++)
    {
      for (int j = 0; j < i; j++)
      {
        const double conv = blockJacobiRotate <T> (A, VT, nx, ny, ld_a, ld_v, i, j);
        if (thread_rank() == 0 && conv > epi)
        { iter = true; }
      }
    }

    __syncthreads();
  }
  return iter_counter;
}



#endif