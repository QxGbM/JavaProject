
#ifndef _DEV_DENSE_FUNCS_CUH
#define _DEV_DENSE_FUNCS_CUH

#include <pspl.cuh>

template <class T>
/* A convinient call to copy from shared memory to global or vice versa. Reading "from" in row major. */
__device__ void matrixCopy_fromRM (const T * __restrict__ from, T * __restrict__ to, const int nx, const int ny, const int ld_from, const int ld_to, const bool transpose)
{
  for (int i = thread_rank(); i < nx * ny; i += block_dim())
  {
    if (transpose)
    { 
      const int row = i / ny, col = i - row * ny;
      to[col * ld_to + row] = from[row * ld_from + col]; 
    }
    else
    { 
      const int row = i / nx, col = i - row * nx;
      to[row * ld_to + col] = from[row * ld_from + col]; 
    }
  }
}

template <class T>
/* A convinient call to copy from shared memory to global or vice versa. Reading "to" in row major. */
__device__ void matrixCopy_toRM (const T * __restrict__ from, T * __restrict__ to, const int nx, const int ny, const int ld_from, const int ld_to, const bool transpose)
{
  for (int i = thread_rank(); i < nx * ny; i += block_dim())
  {
    if (transpose)
    { 
      const int row = i / ny, col = i - row * ny;
      to[row * ld_to + col] = from[col * ld_from + row]; 
    }
    else
    { 
      const int row = i / nx, col = i - row * nx;
      to[row * ld_to + col] = from[row * ld_from + col]; 
    }
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
/* An overloaded version of GEMM, specialized in substraction and k = 1. */
__device__ void blockDenseGemm_K1_RM_Sub (T * __restrict__ M, const T * __restrict__ A, const T * __restrict__ B, const int m, const int n,
  const int ld_m, const int ld_a, const int ld_b, const bool a_T, const bool b_T, T * __restrict__ shm)
{
  matrixCopy_fromRM <T> (B, &shm[0], n, 1, ld_b, n, b_T);
  __syncthreads();

  for (int i = thread_rank(); i < m * n; i += block_dim())
  {
    const int row = i / n, col = i - row * n;
    M[row * ld_m + col] -= (a_T ? A[row] : A[row * ld_a]) * shm[col];
  }
}

template <class T>
/* General Matrix multiplication. M (m by n) = alpha * A (m by k) * B (k by n) + beta * old_M. */
__device__ void blockDenseGemm_shm (const T alpha, const T beta, T * __restrict__ M, const T * __restrict__ A, const T * __restrict__ B, 
  const int m, const int n, const int k, const int ld_m, const int ld_a, const int ld_b, const bool a_T, const bool b_T, T * __restrict__ shm, const int shm_size)
{
  const int t_id = thread_rank(), tb_size = block_dim(), step = shm_size / (m + n);

  if (step == 0)
  {
    for (int i0 = t_id; i0 < m * n; i0 += tb_size)
    {
      const int row = i0 / n, col = i0 - row * n;
      T accum = (beta == 0) ? 0. : beta * M[row * ld_m + col];

      if (alpha != 0.) for (int i1 = 0; i1 < k; i1++)
      { accum += alpha * (a_T ? A[i1 * ld_a + row] : A[row * ld_a + i1]) * (b_T ? B[col * ld_b + i1] : B[i1 * ld_b + col]); }

      M[row * ld_m + col] = accum;
    }
    __syncthreads();
  }
  else
  {
    T * shm_B = &shm[step * m];

    for (int i0 = 0; i0 < k; i0 += step)
    {
      const int k_ = (k - i0 > step) ? step : k - i0;
      matrixCopy_fromRM <T> (&A[a_T ? i0 * ld_a: i0], shm, k_, m, ld_a, k_, a_T);
      matrixCopy_fromRM <T> (&B[b_T ? i0 : i0 * ld_b], shm_B, k_, n, ld_b, k_, !b_T);
      __syncthreads();

      for (int i1 = t_id; i1 < m * n; i1 += tb_size)
      {
        const int row = i1 / n, col = i1 - row * n;
        T accum = i0 ? M[row * ld_m + col] : ((beta == 0.) ? 0. : beta * M[row * ld_m + col]);

        if (alpha != 0.) for (int i2 = 0; i2 < k_; i2++)
        { accum += alpha * shm[row * k_ + i2] * shm_B[col * k_ + i2]; }

        M[row * ld_m + col] = accum;
      }
      __syncthreads();

    }
  }

}

template <class T>
/* General Matrix multiplication with 3 matrices. M (m by n) = alpha * A (m by k) * B (k by l) * C (l by n) + beta * old_M. */
__device__ void blockDenseGemm_3x_shm (const T alpha, const T beta, T * __restrict__ M, const T * __restrict__ A, const T * __restrict__ B, 
  const T * __restrict__ C, const int m, const int n, const int k, const int l, const int ld_m, const int ld_a, const int ld_b, const int ld_c, 
  const bool a_T, const bool b_T, const bool c_T, T * __restrict__ shm, const int shm_size)
{
  const int t_id = thread_rank(), tb_size = block_dim(), step = shm_size / (m + n);
  T * shm_C = &shm[step * m];

  if (step == 0)
  {
    for (int i0 = t_id; i0 < m * n; i0 += tb_size)
    {
      const int row = i0 / n, col = i0 - row * n;
      T accum = (beta == 0.) ? 0. : beta * M[row * ld_m + col];

      if (alpha != 0.) for (int i1 = 0; i1 < k; i1++) for (int i2 = 0; i2 < l; i2++)
      { accum += alpha * (a_T ? A[i1 * ld_a + row] : A[row * ld_a + i1]) * (b_T ? B[i2 * ld_b + i1] : B[i1 * ld_b + i2]) * (c_T ? C[col * ld_c + i2] : C[i2 * ld_c + col]); }

      M[row * ld_m + col] = accum;
    }
    __syncthreads();
  }
  else if (k * n * (m + l) <= m * l * (k + n))
  {
    for (int i0 = 0; i0 < k; i0 += step)
    {
      const int k_ = (k - i0 > step) ? step : k - i0;

      blockDenseGemm_shm <T> (1., 0., shm_C, C, &B[b_T ? i0 : i0 * ld_b], n, k_, l, k_, ld_c, ld_b, !c_T, !b_T, shm, step * m);
      matrixCopy_fromRM <T> (&A[a_T ? i0 * ld_a : i0], shm, k_, m, ld_a, k_, a_T);
      __syncthreads();

      for (int i1 = t_id; i1 < m * n; i1 += tb_size)
      {
        const int row = i1 / n, col = i1 - row * n;
        T accum = i0 ? M[row * ld_m + col] : ((beta == 0.) ? 0. : beta * M[row * ld_m + col]);

        if (alpha != 0.) for (int i2 = 0; i2 < k_; i2++)
        { accum += alpha * shm[row * k_ + i2] * shm_C[col * k_ + i2]; }

        M[row * ld_m + col] = accum;
      }
      __syncthreads();
    }
  }
  else
  {
    for (int i0 = 0; i0 < l; i0 += step)
    {
      const int l_ = (l - i0 > step) ? step : l - i0;

      blockDenseGemm_shm <T> (1., 0., shm, A, &B[b_T ? i0 * ld_b : i0], m, l_, k, l_, ld_a, ld_b, a_T, b_T, shm_C, step * n);
      matrixCopy_fromRM <T> (&C[c_T ? i0 : i0 * ld_c], shm_C, l_, n, ld_c, l_, !c_T);
      __syncthreads();

      for (int i1 = t_id; i1 < m * n; i1 += tb_size)
      {
        const int row = i1 / n, col = i1 - row * n;
        T accum = i0 ? M[row * ld_m + col] : ((beta == 0.) ? 0. : beta * M[row * ld_m + col]);

        if (alpha != 0.) for (int i2 = 0; i2 < l_; i2++)
        { accum += alpha * shm[row * l_ + i2] * shm_C[col * l_ + i2]; }

        M[row * ld_m + col] = accum;
      }
      __syncthreads();
    }
  }

}

template <class T>
/* General Matrix multiplication with 4 matrices. M (m by n) = alpha * A (m by k) * B (k by l) * C (l by o) * D (o by n) + beta * old_M. */
__device__ void blockDenseGemm_4x_shm (const T alpha, const T beta, T * __restrict__ M, const T * __restrict__ A, const T * __restrict__ B, 
  const T * __restrict__ C, const T * __restrict__ D, const int m, const int n, const int k, const int l, const int o, 
  const int ld_m, const int ld_a, const int ld_b, const int ld_c, const int ld_d, const bool a_T, const bool b_T, const bool c_T, const bool d_T, 
  T * __restrict__ shm, const int shm_size)
{
  const int t_id = thread_rank(), tb_size = block_dim(), step = shm_size / (m + n);
  T * shm_D = &shm[step * m];

  if (step == 0)
  {
    for (int i0 = t_id; i0 < m * n; i0 += tb_size)
    {
      const int row = i0 / n, col = i0 - row * n;
      T accum = (beta == 0.) ? 0. : beta * M[row * ld_m + col];

      if (alpha != 0.) for (int i1 = 0; i1 < k; i1++) for (int i2 = 0; i2 < l; i2++) for (int i3 = 0; i3 < o; i3++)
      { accum += alpha * (a_T ? A[i1 * ld_a + row] : A[row * ld_a + i1]) * (b_T ? B[i2 * ld_b + i1] : B[i1 * ld_b + i2]) * 
        (c_T ? C[i3 * ld_c + i2] : C[i2 * ld_c + i3]) * (d_T ? D[col * ld_d + i3] : D[i3 * ld_d + col]); }

      M[row * ld_m + col] = accum;
    }
    __syncthreads();
  }
  else if ((m <= k && m <= l) || (o <= k && o <= l))
  {
    for (int i0 = 0; i0 < o; i0 += step)
    {
      const int o_ = (o - i0 > step) ? step : o - i0;

      blockDenseGemm_3x_shm <T> (1., 0., shm, A, B, &C[c_T ? i0 * ld_c : i0], m, o_, k, l, o_, ld_a, ld_b, ld_c, a_T, b_T, c_T, shm_D, step * n);
      matrixCopy_fromRM <T> (&D[d_T ? i0 : i0 * ld_d], shm_D, o_, n, ld_d, o_, !d_T);
      __syncthreads();

      for (int i1 = t_id; i1 < m * n; i1 += tb_size)
      {
        const int row = i1 / n, col = i1 - row * n;
        T accum = i0 ? M[row * ld_m + col] : ((beta == 0.) ? 0. : beta * M[row * ld_m + col]);

        if (alpha != 0.) for (int i2 = 0; i2 < o_; i2++)
        { accum += alpha * shm[row * o_ + i2] * shm_D[col * o_ + i2]; }

        M[row * ld_m + col] = accum;
      }
      __syncthreads();
    }
  }
  else if ((n <= l && n <= o) || (k <= o && k <= l))
  {
    for (int i0 = 0; i0 < k; i0 += step)
    {
      const int k_ = (k - i0 > step) ? step : k - i0;

      blockDenseGemm_3x_shm <T> (1., 0., shm_D, D, C, &B[b_T ? i0 : i0 * ld_b], n, k_, o, l, k_, ld_d, ld_c, ld_b, !d_T, !c_T, !b_T, shm, step * m);
      matrixCopy_fromRM <T> (&A[a_T ? i0 * ld_a : i0], shm, k_, m, ld_a, k_, a_T);
      __syncthreads();

      for (int i1 = t_id; i1 < m * n; i1 += tb_size)
      {
        const int row = i1 / n, col = i1 - row * n;
        T accum = i0 ? M[row * ld_m + col] : ((beta == 0.) ? 0. : beta * M[row * ld_m + col]);

        if (alpha != 0.) for (int i2 = 0; i2 < k_; i2++)
        { accum += alpha * shm[row * k_ + i2] * shm_D[col * k_ + i2]; }

        M[row * ld_m + col] = accum;
      }
      __syncthreads();
    }
  }
  else
  {
    for (int i0 = 0; i0 < l; i0 += step)
    {
      const int l_ = (l - i0 > step) ? step : l - i0;

      if (n <= m)
      {
        blockDenseGemm_shm <T> (1., 0., shm_D, D, &C[c_T ? i0 : i0 * ld_c], n, l_, o, l_, ld_d, ld_c, !d_T, !c_T, shm, step * m);
        blockDenseGemm_shm <T> (1., 0., shm, A, &B[b_T ? i0 * ld_b : i0], m, l_, k, l_, ld_a, ld_b, a_T, b_T, nullptr, 0);
      }
      else
      {
        blockDenseGemm_shm <T> (1., 0., shm, A, &B[b_T ? i0 * ld_b : i0], m, l_, k, l_, ld_a, ld_b, a_T, b_T, shm_D, step * n);
        blockDenseGemm_shm <T> (1., 0., shm_D, D, &C[c_T ? i0 : i0 * ld_c], n, l_, o, l_, ld_d, ld_c, !d_T, !c_T, nullptr, 0);
      }

      for (int i1 = t_id; i1 < m * n; i1 += tb_size)
      {
        const int row = i1 / n, col = i1 - row * n;
        T accum = i0 ? M[row * ld_m + col] : ((beta == 0.) ? 0. : beta * M[row * ld_m + col]);

        if (alpha != 0.) for (int i2 = 0; i2 < l_; i2++)
        { accum += alpha * shm[row * l_ + i2] * shm_D[col * l_ + i2]; }

        M[row * ld_m + col] = accum;
      }
      __syncthreads();
    }
  }
}

template <class T>
/* General Matrix multiplication with 5 matrices. M (m by n) = alpha * A (m by k) * B (k by l) * C (l by o) * D (o by p) * E (p by n) + beta * old_M. */
__device__ void blockDenseGemm_5x_shm (const T alpha, const T beta, T * __restrict__ M, const T * __restrict__ A, const T * __restrict__ B, 
  const T * __restrict__ C, const T * __restrict__ D, const T * __restrict__ E, const int m, const int n, const int k, const int l, const int o, const int p, 
  const int ld_m, const int ld_a, const int ld_b, const int ld_c, const int ld_d, const int ld_e, 
  const bool a_T, const bool b_T, const bool c_T, const bool d_T, const bool e_T, T * __restrict__ shm, const int shm_size)
{
  const int t_id = thread_rank(), tb_size = block_dim(), step = shm_size / (m + n);
  T * shm_E = &shm[step * m];

  if (step == 0)
  {
    for (int i0 = t_id; i0 < m * n; i0 += tb_size)
    {
      const int row = i0 / n, col = i0 - row * n;
      T accum = (beta == 0.) ? 0. : beta * M[row * ld_m + col];

      if (alpha != 0.) for (int i1 = 0; i1 < k; i1++) for (int i2 = 0; i2 < l; i2++) for (int i3 = 0; i3 < o; i3++) for (int i4 = 0; i4 < p; i4++)
      { accum += alpha * (a_T ? A[i1 * ld_a + row] : A[row * ld_a + i1]) * (b_T ? B[i2 * ld_b + i1] : B[i1 * ld_b + i2]) * 
        (c_T ? C[i3 * ld_c + i2] : C[i2 * ld_c + i3]) * (d_T ? D[i4 * ld_d + i3] : D[i3 * ld_d + i4]) * (e_T ? E[col * ld_e + i4] : E[i4 * ld_e + col]); }

      M[row * ld_m + col] = accum;
    }
    __syncthreads();
  }
  else if ((m <= k && m <= l && m <= o) || (p <= k && p <= l && p <= o))
  {
    for (int i0 = 0; i0 < p; i0 += step)
    {
      const int p_ = (p - i0 > step) ? step : p - i0;

      blockDenseGemm_4x_shm <T> (1., 0., shm, A, B, C, &D[d_T ? i0 * ld_d : i0], m, p_, k, l, o, p_, ld_a, ld_b, ld_c, ld_d, a_T, b_T, c_T, d_T, shm_E, step * n);
      matrixCopy_fromRM <T> (&E[e_T ? i0 : i0 * ld_e], shm_E, p_, n, ld_e, p_, !e_T);
      __syncthreads();

      for (int i1 = t_id; i1 < m * n; i1 += tb_size)
      {
        const int row = i1 / n, col = i1 - row * n;
        T accum = i0 ? M[row * ld_m + col] : ((beta == 0.) ? 0. : beta * M[row * ld_m + col]);

        if (alpha != 0.) for (int i2 = 0; i2 < p_; i2++)
        { accum += alpha * shm[row * p_ + i2] * shm_E[col * p_ + i2]; }

        M[row * ld_m + col] = accum;
      }
      __syncthreads();
    }
  }
  else if ((n <= l && n <= o && n <= p) || (k <= l && k <= o && k <= p))
  {
    for (int i0 = 0; i0 < k; i0 += step)
    {
      const int k_ = (k - i0 > step) ? step : k - i0;

      blockDenseGemm_4x_shm <T> (1., 0., shm_E, E, D, C, &B[b_T ? i0 : i0 * ld_b], n, k_, p, o, l, k_, ld_e, ld_d, ld_c, ld_b, !e_T, !d_T, !c_T, !b_T, shm, step * m);
      matrixCopy_fromRM <T> (&A[a_T ? i0 * ld_a : i0], shm, k_, m, ld_a, k_, a_T);
      __syncthreads();

      for (int i1 = t_id; i1 < m * n; i1 += tb_size)
      {
        const int row = i1 / n, col = i1 - row * n;
        T accum = i0 ? M[row * ld_m + col] : ((beta == 0.) ? 0. : beta * M[row * ld_m + col]);

        if (alpha != 0.) for (int i2 = 0; i2 < k_; i2++)
        { accum += alpha * shm[row * k_ + i2] * shm_E[col * k_ + i2]; }

        M[row * ld_m + col] = accum;
      }
      __syncthreads();
    }
  }
  else if (l <= o)
  {
    for (int i0 = 0; i0 < l; i0 += step)
    {
      const int l_ = (l - i0 > step) ? step : l - i0;

      blockDenseGemm_3x_shm <T> (1., 0., shm_E, E, D, &C[c_T ? i0 : i0 * ld_c], n, l_, p, o, l_, ld_e, ld_d, ld_c, !e_T, !d_T, !c_T, shm, step * m);
      blockDenseGemm_shm <T> (1., 0., shm, A, &B[b_T ? i0 * ld_b : i0], m, l_, k, l_, ld_a, ld_b, a_T, b_T, nullptr, 0);

      for (int i1 = t_id; i1 < m * n; i1 += tb_size)
      {
        const int row = i1 / n, col = i1 - row * n;
        T accum = i0 ? M[row * ld_m + col] : ((beta == 0.) ? 0. : beta * M[row * ld_m + col]);

        if (alpha != 0.) for (int i2 = 0; i2 < l_; i2++)
        { accum += alpha * shm[row * l_ + i2] * shm_E[col * l_ + i2]; }

        M[row * ld_m + col] = accum;
      }
      __syncthreads();
    }
  }
  else
  {
    for (int i0 = 0; i0 < o; i0 += step)
    {
      const int o_ = (o - i0 > step) ? step : o - i0;

      blockDenseGemm_3x_shm <T> (1., 0., shm, A, B, &C[c_T ? i0 * ld_c : i0], m, o_, k, l, o_, ld_a, ld_b, ld_c, a_T, b_T, c_T, shm_E, step * n);
      blockDenseGemm_shm <T> (1., 0., shm_E, E, &D[d_T ? i0 : i0 * ld_d], n, o_, p, o_, ld_e, ld_d, !e_T, !d_T, nullptr, 0);

      for (int i1 = t_id; i1 < m * n; i1 += tb_size)
      {
        const int row = i1 / n, col = i1 - row * n;
        T accum = i0 ? M[row * ld_m + col] : ((beta == 0.) ? 0. : beta * M[row * ld_m + col]);

        if (alpha != 0.) for (int i2 = 0; i2 < o_; i2++)
        { accum += alpha * shm[row * o_ + i2] * shm_E[col * o_ + i2]; }

        M[row * ld_m + col] = accum;
      }
      __syncthreads();
    }
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
__device__ int blockReduceMax_Index (const T * __restrict__ M, const int n, int * __restrict__ shm)
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
__device__ void blockSwapRows (T * __restrict__ row1, T * __restrict__ row2, const int n)
{
  for (int i = thread_rank(); i < n; i += block_dim())
  { const T t = row1[i]; row1[i] = row2[i]; row2[i] = t; }
}

template <class T> 
/* Exchange col1[0] with col2[0], col1[1] with col2[1], ... col1[n-1] with col2[n-1]. */
__device__ void blockSwapColumns (T * __restrict__ col1, T * __restrict__ col2, const int n, const int ld)
{
  for (int i = thread_rank(); i < n; i += block_dim())
  { const T t = col1[i * ld]; col1[i * ld] = col2[i * ld]; col2[i * ld] = t; }
}

template <class T> 
/* Using a group of threads to apply pivot the pivot swaps to the matrix. Recover flag retrieves original matrix. Utilizes L1. */
__device__ void blockApplyPivot (T * __restrict__ M, const int * __restrict__ p, const int nx, const int ny, const int ld, const bool recover, 
  T * __restrict__ shm, const int shm_size)
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

    matrixCopy_toRM(&shm[0], &M[n], cols, ny, cols, ld, false);
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
__device__ void blockDenseGetrf_shm (T * __restrict__ M, int * __restrict__ p, const int nx, const int ny, const int ld, T * __restrict__ shm)
{
  if (p != nullptr) { resetPivot(p, ny); }

  for (int i = 0; i < nx && i < ny; i++)
  {
    matrixCopy_fromRM <T> (&M[i * ld + i], &shm[0], 1, ny - i, ld, 1, false);
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

    blockDenseGemm_K1_RM_Sub <T> (&M[(i + 1) * ld + (i + 1)], &shm[1], &M[i * ld + (i + 1)], ny - (i + 1), nx - (i + 1), ld, 1, ld, false, false, &shm[ny - i]);

    matrixCopy_toRM <T> (&shm[0], &M[i * ld + i], 1, ny - i, 1, ld, false);
    __syncthreads();
  }

}

template <class T>
/* L is ny_l x nx_l lower triangular and unit diagonal, B is ny_l by nx_b, solves L x X = B, overwrites X in B. Utilizes L1 cache. */
__device__ void blockDenseTrsmL_shm (T * __restrict__ B, const T * __restrict__ L, const int nx_b, const int ny_b, const int nx_l, 
  const int ld_b, const int ld_l, const bool B_T, T * __restrict__ shm, const int shm_size)
{
  for (int i = 0; i < nx_l && i + 1 < ny_b; i++)
  { 
    blockDenseGemm_K1_RM_Sub <T> (&B[(i + 1) * ld_b], &L[(i + 1) * ld_l + i], &B[i * ld_b], ny_b - (i + 1), nx_b, ld_b, ld_l, ld_b, false, false, shm);
    __syncthreads();
  }
}

 
template <class T>
/* U is ny_u x nx_u upper triangular and not unit diagonal, B is ny_b by nx_u, solves X x U = B, overwrites X in B. Utilizes L1 cache. */
 __device__ void blockDenseTrsmR_shm (T * __restrict__ B, const T * __restrict__ U, const int nx_b, const int ny_b, const int ny_u, 
   const int ld_b, const int ld_u, const bool B_T, T * __restrict__ shm, const int shm_size)
{
   const int rows_block = shm_size / nx_b;
   int rows_index = 0;

   while (rows_index < ny_b)
   {
     const int rows_remaining = ny_b - rows_index, rows = (rows_remaining > rows_block) ? rows_block : rows_remaining;

     matrixCopy_fromRM <T> (&B[B_T ? rows_index : rows_index * ld_b], shm, nx_b, rows, ld_b, nx_b, B_T);
     __syncthreads();

     for (int i = 0; i < nx_b && i < ny_u; i++)
     {
       for (int j = warp_rank(); j < rows; j += num_warps())
       {
         if (lane_rank() == 0)
         { shm[j * nx_b + i] /= U[i * ld_u + i]; }
         __syncwarp();

         for (int k = lane_rank() + i + 1; k < nx_b; k += warpSize)
         { shm[j * nx_b + k] -= shm[j * nx_b + i] * U[i * ld_u + k]; }
       }
       __syncthreads();
     }

     matrixCopy_toRM <T> (shm, &B[B_T ? rows_index : rows_index * ld_b], nx_b, rows, nx_b, ld_b, B_T);
     rows_index += rows;
     __syncthreads();
   }

}


#endif