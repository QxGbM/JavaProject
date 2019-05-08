#ifndef _DEV_LOW_RANK_FUNCS_CUH
#define _DEV_LOW_RANK_FUNCS_CUH

#include <pspl.cuh>

__constant__ double seed[_RND_SEED_LENGTH];

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

template <class T> 
__device__ bool blockJacobiSVD_iter (T * __restrict__ UxS, T * __restrict__ VT, const int nx, const int ny, const int ld_UxS, const int ld_VT, T * __restrict__ shm, const T epi)
{
  bool iter = false;

  for (int step = nx / 2; step > 0; step /= 2) for (int offset = 0; offset < step; offset++)
  {
    for (int i = thread_rank(); i < 2 * nx; i += block_dim())
    { shm[i] = 0.; }
    __syncthreads();

    for (int row = 0; row < ny; row++)
    {
      const int row_UxS = row * ld_UxS;
      for (int col = thread_rank(); col < nx; col += block_dim())
      {
        const T e = UxS[row_UxS + col];
        shm[col] += e * e;
        const int lane = col % (2 * step);
        if (lane < step)
        { 
          if (lane + offset >= step)
          { shm[nx + col] += e * UxS[row_UxS + col + offset]; }
          else
          { shm[nx + col] += e * UxS[row_UxS + col + step + offset]; }
        }
      }
    }
    __syncthreads();

    for (int col = warp_rank(); col < nx; col += num_warps())
    {
      if (shm[nx + col] > epi || shm[nx + col] < -epi)
      {
        double sine, cosine;
        const int col2 = col + offset + ((col % step + offset >= step) ? 0 : step);
        const bool swap = shm[col] < shm[col2];
        iter = true;

        if (lane_rank() == 0)
        {
          const double torque = (double) ((shm[col2] - shm[col]) / (shm[nx + col] * 2.));
          const double sign_torque = (double) signbit(torque) * -2 + 1;
          const double tangent = sign_torque / (fabs(torque) + sqrt(1. + torque * torque));
          cosine = rsqrt(1. + tangent * tangent);
          sine = cosine * tangent;
        }
        __syncwarp();

        cosine = __shfl_sync(0xffffffff, cosine, 0, warpSize);
        sine = __shfl_sync(0xffffffff, sine, 0, warpSize);

        for (int row = lane_rank(); row < ny; row += warpSize)
        {
          const int row_UxS = row * ld_UxS;
          const T e1 = swap ? UxS[row_UxS + col2] : UxS[row_UxS + col], e2 = swap ? UxS[row_UxS + col] : UxS[row_UxS + col2];
          UxS[row_UxS + col] = cosine * e1 - sine * e2;
          UxS[row_UxS + col2] = sine * e1 + cosine * e2;
        }

        for (int row = lane_rank(); row < nx; row += warpSize)
        {
          const int row_VT = row * ld_VT;
          const T e3 = swap ? VT[row_VT + col2] : VT[row_VT + col], e4 = swap ? VT[row_VT + col] : VT[row_VT + col2];
          VT[row_VT + col] = cosine * e3 - sine * e4;
          VT[row_VT + col2] = sine * e3 + cosine * e4;
        }
      }
    }
    __syncthreads();
  }

  return iter;
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

template <class T> __device__ void blockDenseGeqrf (T * __restrict__ M, const int nx, const int ny, const int ld, T * __restrict__ shm)
{
  for (int i = 0; i < nx; i++)
  {
    for (int col = i + thread_rank(); col < nx; col += block_dim())
    { shm[col] = 0; }
    __syncthreads();

    for (int row = 0; row < ny; row++) for (int col = i + thread_rank(); col < nx; col += block_dim())
    { shm[col] += M[row * ld + i] * M[row * ld + col]; }
    __syncthreads();

    if (thread_rank() == 0)
    { shm[i] = rsqrt(shm[i]); }
    __syncthreads();

    for (int col = i + thread_rank() + 1; col < nx; col += block_dim())
    { shm[col] *= shm[i]; }
    __syncthreads();

    for (int row = 0; row < ny; row++) 
    {
      if (thread_rank() == 0)
      { shm[nx] = M[row * ld + i] * shm[i]; }
      __syncthreads();

      if (thread_rank() == 0)
      { M[row * ld + i] = shm[nx]; }
      for (int col = i + thread_rank() + 1; col < nx; col += block_dim())
      { M[row * ld + col] = M[row * ld + col] - shm[col] * shm[nx]; }
      __syncthreads();
    }
  }

}

template <class T> 
__device__ int blockRandomizedSVD (T * __restrict__ A, T * __restrict__ VT, const int nx, const int ny, const int ld_a, const int ld_v, 
  const int k, const double epi, const int iter_limit, T * __restrict__ shm, const int shm_size)
{
  const int P = (2 * k > nx) ? nx : 2 * k;

  T * X, ** X_ptr = (T **) &shm[0], *B, ** B_ptr = (T **) &shm[1];
  if (thread_rank() == 0)
  { X = new T[ny * P]; *X_ptr = X; B = new T[P * nx]; *B_ptr = B; }
  __syncthreads();

  X = *X_ptr; B = *B_ptr;
  blockDenseGemm_shm (1., 0., X, A, seed, ny, P, nx, P, ld_a, P, false, false, shm, shm_size);
  blockDenseGeqrf (X, P, ny, P, shm);

  blockDenseGemm_shm (1., 0., B, X, A, P, nx, ny, nx, P, ld_a, true, false, shm, shm_size);

  int * iter = (int *) &shm[0], *loop_counter = (int *) &shm[1];
  if (thread_rank() == 0)
  { *iter = 1; *loop_counter = 0; }
  __syncthreads();

  while (*iter && *loop_counter < iter_limit)
  {
    if (thread_rank() == 0)
    { *iter = 0; (*loop_counter)++; }

    if (blockJacobiSVD_iter(B, VT, nx, P, nx, ld_v, &shm[2], epi))
    { *iter = 1; }
    __syncthreads();
  }
  int loops = *loop_counter;
  
  blockDenseGemm_shm (1., 0., A, X, B, ny, nx, P, ld_a, P, nx, false, false, shm, shm_size);
  if (thread_rank() == 0)
  { delete X; delete B; }

  return loops;

}


#endif