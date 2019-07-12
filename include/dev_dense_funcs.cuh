
#pragma once
#ifndef _DEV_DENSE_FUNCS_CUH
#define _DEV_DENSE_FUNCS_CUH

#include <pspl.cuh>

template <class T, class vecT, int vec_size>
/* A convinient call to copy from shared memory to global or vice versa. Reading "from" in row major. */
__device__ __forceinline__ void matrixCopy (const T * __restrict__ from, T * __restrict__ to, const int nx_to, const int ny_to, const int ld_from, const int ld_to)
{
  const int w_id = warp_rank(), l_id = lane_rank(), n_wp = num_warps();
  const int iter = nx_to / vec_size, last_start = iter * vec_size, last = nx_to - last_start;

  for (int row = w_id; row < ny_to; row += n_wp)
  {
    T * to_row = &to[row * ld_to];
    const T * from_row = &from[row * ld_from];

    for (int col = l_id; col < iter; col += warpSize)
    { reinterpret_cast <vecT *> (to_row)[col] = reinterpret_cast <const vecT *> (from_row)[col]; }
  }

  if (last > 0)
  for (int row = w_id; row < ny_to; row += n_wp)
  {
    T * to_row = &to[row * ld_to + last_start];
    const T * from_row = &from[row * ld_from + last_start];

    for (int col = l_id; col < last; col += warpSize)
    { to_row[col] = from_row[col]; }
  }

}

template <class T, class vecT, int vec_size>
/* A convinient call to copy from shared memory to global or vice versa. Reading "from" in row major. */
__device__ __forceinline__ int matrixCopy_keepT (const T * __restrict__ from, T * __restrict__ to, const int nx_from, const int ny_from, const int ld_from, const bool transpose)
{
  const int w_id = warp_rank(), l_id = lane_rank(), n_wp = num_warps();

  const int nx_real = transpose ? ny_from : nx_from, ny_real = transpose ? nx_from : ny_from;
  const int ld_to = ((nx_real + vec_size - 1) / vec_size) * vec_size;

  const int iter = nx_real / vec_size, last_start = iter * vec_size, last = nx_real - iter * vec_size;

  for (int row = w_id; row < ny_real; row += n_wp)
  {
    T * to_row = &to[row * ld_to];
    const T * from_row = &from[row * ld_from];

    for (int col = l_id; col < iter; col += warpSize)
    { reinterpret_cast <vecT *> (to_row)[col] = reinterpret_cast <const vecT *> (from_row)[col]; }
  }

  if (last > 0)
  for (int row = w_id; row < ny_real; row += n_wp)
  {
    T * to_row = &to[row * ld_to + last_start];
    const T * from_row = &from[row * ld_from + last_start];

    for (int col = l_id; col < last; col += warpSize)
    { to_row[col] = from_row[col]; }
  }

  return ld_to;

}


template <class T, class vecT, int vec_size> 
/* LU decomposition of matrix of ny by nx. */
__device__ __forceinline__ void DenseGetrf (T * M, const int nx, const int ny, const int ld)
{
  const int w_id = warp_rank(), l_id = lane_rank(), n_wp = num_warps(), min_n = nx > ny ? ny : nx;

  T left, vec0[vec_size], vec1[vec_size];
  
  for (int i = 0; i < min_n; i ++)
  {
    const int x_start = i + 1, x_n = nx - x_start;

    const int iter = x_n / vec_size, last = x_n - iter * vec_size, align = x_start + last;

    T * M_top = &M[i * ld], * M_top_align = &M_top[align];

    for (int row = w_id + i + 1; row < ny; row += n_wp)
    {
      T * M_row = &M[row * ld], * M_row_align = &M_row[align];

      if (l_id == 0)
      { M_row[i] = left = M_row[i] / M_top[i]; }

      left = __shfl_sync (0xffffffff, - left, 0, warpSize);

      if (last > 0)
      for (int col = x_start + l_id; col < align; col += warpSize)
      { M_row[col] = fma (M_top[col], left, M_row[col]); }

      for (int col = l_id; col < iter; col += warpSize)
      {
        reinterpret_cast <vecT *> (vec0)[0] = reinterpret_cast <vecT *> (M_top_align)[col];
        reinterpret_cast <vecT *> (vec1)[0] = reinterpret_cast <vecT *> (M_row_align)[col];

        #pragma unroll
        for (int i1 = 0; i1 < vec_size; i1++)
        { vec1[i1] = fma (vec0[i1], left, vec1[i1]); }

        reinterpret_cast <vecT *> (M_row_align)[col] = reinterpret_cast <vecT *> (vec1)[0];
      }

    }
    __syncthreads();
  }

}

template <class T, class vecT, int vec_size>
/* L is ny_l x nx_l lower triangular and unit diagonal, B is ny_l by nx_b, solves L x X = B, overwrites X in B. */
__device__ __forceinline__ void DenseTrsmL (T * __restrict__ B, const T * __restrict__ L, const int nx_b, const int ny_b, const int nx_l, const int ld_b, const int ld_l)
{
  const int w_id = warp_rank(), l_id = lane_rank(), n_wp = num_warps(), min_n = nx_l > ny_b ? ny_b : nx_l;

  const int iter = nx_b / vec_size, serial_start = iter * vec_size, last = nx_b - serial_start;
  T vec0[vec_size], vec1[vec_size];

  for (int i = 0; i < min_n; i ++)
  {
    for (int row = w_id + i + 1; row < ny_b; row += n_wp)
    {
      T left = - L[row * ld_l + i], * B_top = &B[i * ld_b], * B_row = &B[row * ld_b];

      for (int col = l_id; col < iter; col += warpSize)
      {
        reinterpret_cast <vecT *> (vec0)[0] = reinterpret_cast <vecT *> (B_top)[col];
        reinterpret_cast <vecT *> (vec1)[0] = reinterpret_cast <vecT *> (B_row)[col];

        #pragma unroll
        for (int i1 = 0; i1 < vec_size; i1++)
        { vec1[i1] = fma (vec0[i1], left, vec1[i1]); }

        reinterpret_cast <vecT *> (B_row)[col] = reinterpret_cast <vecT *> (vec1)[0];
      }

      if (last > 0)
      for (int col = serial_start + l_id; col < nx_b; col += warpSize)
      { B_row[col] = fma (B_top[col], left, B_row[col]); }
    }
    __syncthreads();
  }

}

template <class T, class vecT, int vec_size>
/* U is ny_u x nx_u upper triangular and not unit diagonal, B is ny_b by nx_u, solves X x U = B, overwrites X in B. */
__device__ __forceinline__ void DenseTrsmR (T * __restrict__ B, const T * __restrict__ U, const int nx_b, const int ny_b, const int ny_u, const int ld_b, const int ld_u)
{
  const int w_id = warp_rank(), l_id = lane_rank(), n_wp = num_warps(), min_n = nx_b > ny_u ? ny_u : nx_b;

  T left, vec0[vec_size], vec1[vec_size];

  for (int i = 0; i < min_n; i ++)
  {
    const int i_start = i + 1, x = nx_b - i_start;
    const int iter = x / vec_size, last = x - iter * vec_size, last_start = i_start + last;

    const T * U_top = &U[i * ld_u], * U_top_vec = &U_top[last_start];

    for (int row = w_id; row < ny_b; row += n_wp)
    {
      T * B_row = &B[row * ld_b], * B_row_vec = &B_row[last_start];

      if (l_id == 0)
      { B_row[i] = left = B_row[i] / U_top[i]; }

      left = __shfl_sync (0xffffffff, - left, 0, warpSize);

      if (last > 0)
      for (int col = l_id + i_start; col < last_start; col += warpSize)
      { B_row[col] = fma (left, U_top[col], B_row[col]); }

      for (int col = l_id; col < iter; col += warpSize)
      {
        reinterpret_cast <vecT *> (vec0)[0] = reinterpret_cast <const vecT *> (U_top_vec)[col];
        reinterpret_cast <vecT *> (vec1)[0] = reinterpret_cast <vecT *> (B_row_vec)[col];

        #pragma unroll
        for (int i1 = 0; i1 < vec_size; i1++)
        { vec1[i1] = fma (vec0[i1], left, vec1[i1]); }

        reinterpret_cast <vecT *> (B_row_vec)[col] = reinterpret_cast <vecT *> (vec1)[0];
      }
    }
    __syncthreads();
  }

}

template <class T, class vecT, int vec_size>
/* U is ny_u x nx_u upper triangular and not unit diagonal, B is ny_b by nx_u, solves X x U = B, overwrites X in B. */
__device__ __forceinline__ void DenseTrsmR_transposeB (T * __restrict__ B, const T * __restrict__ U, const int nx_b, const int ny_b, const int ny_u, const int ld_b, const int ld_u)
{
  const int w_id = warp_rank(), l_id = lane_rank(), n_wp = num_warps(), min_n = nx_b > ny_u ? ny_u : nx_b;

  const int iter = ny_b / vec_size, serial_start = iter * vec_size, last = ny_b - serial_start;
  T vec0[vec_size], vec1[vec_size];

  for (int i = 0; i < min_n; i ++)
  {
    T * B_top = &B[i * ld_b];
    const T * U_top = &U[i * ld_u];

    if (w_id == 0)
    {
      T diag = 1. / U_top[i];

      for (int col = l_id; col < iter; col += warpSize)
      { 
        reinterpret_cast <vecT *> (vec1)[0] = reinterpret_cast <vecT *> (B_top)[col];

        #pragma unroll
        for (int i1 = 0; i1 < vec_size; i1++)
        { vec1[i1] *= diag; }

        reinterpret_cast <vecT *> (B_top)[col] = reinterpret_cast <vecT *> (vec1)[0];
      }

      if (last > 0)
      for (int col = l_id + serial_start; col < ny_b; col += warpSize)
      { B_top[col] *= diag; }
    }
    __syncthreads();

    for (int row = w_id + i + 1; row < nx_b; row += n_wp)
    {
      T left = - U_top[row], * B_row = &B[row * ld_b];

      for (int col = l_id; col < iter; col += warpSize)
      { 
        reinterpret_cast <vecT *> (vec0)[0] = reinterpret_cast <vecT *> (B_top)[col];
        reinterpret_cast <vecT *> (vec1)[0] = reinterpret_cast <vecT *> (B_row)[col];

        #pragma unroll
        for (int i1 = 0; i1 < vec_size; i1++)
        { vec1[i1] = fma (left, vec0[i1], vec1[i1]); }

        reinterpret_cast <vecT *> (B_row)[col] = reinterpret_cast <vecT *> (vec1)[0];
      }

      if (last > 0)
      for (int col = l_id + serial_start; col < ny_b; col += warpSize)
      { B_row[col] = fma (left, B_top[col], B_row[col]); }
    }
    __syncthreads();
  }

}

template <class T, class vecT, int vec_size>
/* General Matrix multiplication. M (m by n) = A (m by k) * B (k by n) + old_M. */
__device__ __forceinline__ void DenseGemm (T * __restrict__ M, const T * __restrict__ A, const T * __restrict__ B, const int m, const int n, const int k, 
  const int ld_m, const int ld_a, const int ld_b, const bool a_T, const bool b_T)
{
  const int w_id = warp_rank(), l_id = lane_rank(), n_wp = num_warps();

  int A_step, B_step, A_iter, B_iter;

  if (a_T) { A_step = 1; A_iter = ld_a; } else { A_step = ld_a; A_iter = 1; }
  if (b_T) { B_step = ld_b; B_iter = 1; } else { B_step = 1; B_iter = ld_b; }

  const int iter_m = m / vec_size, iter_n = n / vec_size;
  const int last_m_start = iter_m * vec_size, last_n_start = iter_n * vec_size;
  const int last_m = m - last_m_start, last_n = n - last_n_start;

  const bool b_last_m = last_m - w_id > 0, b_last_n = last_n > 0;

  T thread_a[vec_size], thread_b[vec_size], thread_m[vec_size][vec_size];

  for (int i = 0; i < k; i++)
  {
    const T * A_k = &A[i * A_iter], * B_k = &B[i * B_iter];

    for (int i1 = w_id; i1 < iter_m; i1 += n_wp)
    {
      const int row_start = i1 * vec_size;

      #pragma unroll
      for (int i3 = 0; i3 < vec_size; i3++)
      {
        const int row = row_start + i3;
        thread_a[i3] = A_k[row * A_step];
      }

      for (int i2 = l_id; i2 < iter_n; i2 += warpSize)
      {
        const int col_start = i2 * vec_size;

        #pragma unroll
        for (int i3 = 0; i3 < vec_size; i3++)
        {
          const int row = row_start + i3;
          reinterpret_cast <vecT *> (thread_m[i3])[0] = reinterpret_cast <vecT *> (&M[row * ld_m])[i2];
        }

        #pragma unroll
        for (int i4 = 0; i4 < vec_size; i4++)
        {
          const int col = col_start + i4;
          thread_b[i4] = B_k[col * B_step];
        }

        #pragma unroll
        for (int i3 = 0; i3 < vec_size; i3++)
        {
          #pragma unroll
          for (int i4 = 0; i4 < vec_size; i4++)
          { thread_m[i3][i4] = fma (thread_a[i3], thread_b[i4], thread_m[i3][i4]); }
        }

        #pragma unroll
        for (int i3 = 0; i3 < vec_size; i3++)
        {
          const int row = row_start + i3;
          reinterpret_cast <vecT *> (&M[row * ld_m])[i2] = reinterpret_cast <vecT *> (thread_m[i3])[0];
        }

      }

      if (b_last_n)
      {
        #pragma unroll
        for (int i3 = 0; i3 < vec_size; i3++)
        {
          const int row = row_start + i3;
          T * M_row = &M[row * ld_m + last_n_start];

          for (int col = l_id; col < last_n; col += warpSize)
          { M_row[col] = fma (thread_a[i3], B_k[col * B_step], M_row[col]); }
        }
      }

    }

    if (b_last_m)
    {
      const int row = w_id + last_m_start; const T left = A_k[row * A_step];
      T * M_row = &M[row * ld_m];

      for (int i2 = l_id; i2 < iter_n; i2 += warpSize)
      {
        const int col_start = i2 * vec_size;
        reinterpret_cast <vecT *> (thread_a)[0] = reinterpret_cast <vecT *> (M_row)[i2];

        #pragma unroll
        for (int i4 = 0; i4 < vec_size; i4++)
        {
          const int col = col_start + i4;
          thread_b[i4] = B_k[col * B_step]; 
        }

        #pragma unroll
        for (int i4 = 0; i4 < vec_size; i4++)
        { thread_a[i4] = fma(left, thread_b[i4], thread_a[i4]); }

        reinterpret_cast <vecT *> (M_row)[i2] = reinterpret_cast <vecT *> (thread_a)[0];
      }

      if (b_last_n)
      {
        T * M_row_n = &M_row[last_n_start];

        #pragma unroll
        for (int i3 = 0; i3 < vec_size; i3++)
        {
          for (int col = l_id; col < last_n; col += warpSize)
          { M_row_n[col] = fma (left, B_k[col * B_step], M_row_n[col]); }
        }
      }
    }
  }

}

template <class T, class vecT, int vec_size, int block_dim_m, int block_dim_k>
/* General Matrix multiplication. M (m by n) = alpha * A (m by k) * B (k by n) + beta * old_M. */
__device__ __forceinline__ void blockDenseGemm (const T alpha, const T beta, T * __restrict__ M, const T * __restrict__ A, const T * __restrict__ B,
  const int m, const int n, const int k, const int ld_m, const int ld_a, const int ld_b, const bool a_T, const bool b_T, T * __restrict__ shm)
{
  const int w_id = warp_rank(), l_id = lane_rank(), n_wp = num_warps();

  const int iter_n = n / vec_size;
  const int last_n_start = iter_n * vec_size;
  const int last_n = n - last_n_start;

  const bool b_last_n = last_n > 0;

  T mult = beta / alpha;

  if (beta == 0.)
  {
    T thread_a[vec_size];
    #pragma unroll
    for (int i = 0; i < vec_size; i++)
    { thread_a[i] = 0.; }
    
    vecT zero_vec = reinterpret_cast <vecT *> (thread_a)[0];

    for (int row = w_id; row < m; row += n_wp)
    {
      T * M_row = &M[row * ld_m];

      for (int col = l_id; col < iter_n; col += warpSize)
      { reinterpret_cast <vecT *> (M_row)[col] = zero_vec; }
    }

    if (b_last_n)
    for (int row = w_id; row < m; row += n_wp)
    {
      T * M_row = &M[row * ld_m + last_n_start];

      for (int col = l_id; col < last_n; col += warpSize)
      { M_row[col] = 0.; }
    }
    __syncthreads();
  }
  else if (mult != 1.)
  {
    T thread_a[vec_size];

    for (int row = w_id; row < m; row += n_wp)
    {
      T * M_row = &M[row * ld_m];

      for (int col = l_id; col < iter_n; col += warpSize)
      {
        reinterpret_cast <vecT *> (thread_a)[0] = reinterpret_cast <vecT *> (M_row)[col];

        #pragma unroll
        for (int i = 0; i < vec_size; i++)
        { thread_a[i] *= mult; }

        reinterpret_cast <vecT *> (M_row)[col] = reinterpret_cast <vecT *> (thread_a)[0];
      }
    }

    if (b_last_n)
    for (int row = w_id; row < m; row += n_wp)
    {
      T * M_row = &M[row * ld_m + last_n_start];

      for (int col = l_id; col < last_n; col += warpSize)
      { M_row[col] *= mult; }
    }
    __syncthreads();
  }

  T * shm_a = &shm[block_dim_m * block_dim_m], * shm_b = &shm[block_dim_m * (block_dim_m + block_dim_k)];

  const int iters_m = m / block_dim_m, iters_n = n / block_dim_m, iters_k = k / block_dim_k;
  const int last_m_dim = m - iters_m * block_dim_m, last_n_dim = n - iters_n * block_dim_m;
  const int last_k_dim = k - iters_k * block_dim_k;

  int A_step_r, B_step_r, A_step_c, B_step_c;

  if (a_T) { A_step_r = 1; A_step_c = ld_a; } else { A_step_r = ld_a; A_step_c = 1; }
  if (b_T) { B_step_r = 1; B_step_c = ld_b; } else { B_step_r = ld_b; B_step_c = 1; }

  for (int i0 = 0; i0 < iters_m; i0++)
  {
    const int m_off = i0 * block_dim_m;
    T * M_row = &M[m_off * ld_m]; const T * A_row = &A[m_off * A_step_r];

    for (int i1 = 0; i1 < iters_n; i1++)
    {
      const int n_off = i1 * block_dim_m; const T * B_col = &B[n_off * B_step_c];
      const int ld_0 = matrixCopy_keepT <T, vecT, vec_size> (&M_row[n_off], shm, block_dim_m, block_dim_m, ld_m, false);

      for (int i2 = 0; i2 < iters_k; i2++)
      {
        const int k_off = i2 * block_dim_k; const T * A_k = &A_row[k_off * A_step_c], * B_k = &B_col[k_off * B_step_r];
        const int ld_1 = matrixCopy_keepT <T, vecT, vec_size> (A_k, shm_a, block_dim_k, block_dim_m, ld_a, a_T);
        const int ld_2 = matrixCopy_keepT <T, vecT, vec_size> (B_k, shm_b, block_dim_m, block_dim_k, ld_b, b_T);
        __syncthreads();

        DenseGemm <T, vecT, vec_size> (shm, shm_a, shm_b, block_dim_m, block_dim_m, block_dim_k, ld_0, ld_1, ld_2, a_T, b_T);
        __syncthreads();
      }

      if (last_k_dim > 0)
      {
        const int k_off = iters_k * block_dim_k; const T * A_k = &A_row[k_off * A_step_c], * B_k = &B_col[k_off * B_step_r];
        const int ld_1 = matrixCopy_keepT <T, vecT, vec_size> (A_k, shm_a, last_k_dim, block_dim_m, ld_a, a_T);
        const int ld_2 = matrixCopy_keepT <T, vecT, vec_size> (B_k, shm_b, block_dim_m, last_k_dim, ld_b, b_T);
        __syncthreads();

        DenseGemm <T, vecT, vec_size> (shm, shm_a, shm_b, block_dim_m, block_dim_m, last_k_dim, ld_0, ld_1, ld_2, a_T, b_T);
        __syncthreads();
      }

      matrixCopy <T, vecT, vec_size> (shm, &M_row[n_off], block_dim_m, block_dim_m, ld_0, ld_m);
      __syncthreads();
    }

    if (last_n_dim > 0)
    {
      const int n_off = iters_n * block_dim_m; const T * B_col = &B[n_off * B_step_c];
      const int ld_0 = matrixCopy_keepT <T, vecT, vec_size> (&M_row[n_off], shm, last_n_dim, block_dim_m, ld_m, false);

      for (int i2 = 0; i2 < iters_k; i2++)
      {
        const int k_off = i2 * block_dim_k; const T * A_k = &A_row[k_off * A_step_c], * B_k = &B_col[k_off * B_step_r];
        const int ld_1 = matrixCopy_keepT <T, vecT, vec_size> (A_k, shm_a, block_dim_k, block_dim_m, ld_a, a_T);
        const int ld_2 = matrixCopy_keepT <T, vecT, vec_size> (B_k, shm_b, last_n_dim, block_dim_k, ld_b, b_T);
        __syncthreads();

        DenseGemm <T, vecT, vec_size> (shm, shm_a, shm_b, block_dim_m, last_n_dim, block_dim_k, ld_0, ld_1, ld_2, a_T, b_T);
        __syncthreads();
      }

      if (last_k_dim > 0)
      {
        const int k_off = iters_k * block_dim_k; const T * A_k = &A_row[k_off * A_step_c], * B_k = &B_col[k_off * B_step_r];
        const int ld_1 = matrixCopy_keepT <T, vecT, vec_size> (A_k, shm_a, last_k_dim, block_dim_m, ld_a, a_T);
        const int ld_2 = matrixCopy_keepT <T, vecT, vec_size> (B_k, shm_b, last_n_dim, last_k_dim, ld_b, b_T);
        __syncthreads();

        DenseGemm <T, vecT, vec_size> (shm, shm_a, shm_b, block_dim_m, last_n_dim, last_k_dim, ld_0, ld_1, ld_2, a_T, b_T);
        __syncthreads();
      }

      matrixCopy <T, vecT, vec_size> (shm, &M_row[n_off], last_n_dim, block_dim_m, ld_0, ld_m);
      __syncthreads();
    }
  }

  if (last_m_dim > 0)
  {
    const int m_off = iters_m * block_dim_m;
    T * M_row = &M[m_off * ld_m]; const T * A_row = &A[m_off * A_step_r];

    for (int i1 = 0; i1 < iters_n; i1++)
    {
      const int n_off = i1 * block_dim_m; const T * B_col = &B[n_off * B_step_c];
      const int ld_0 = matrixCopy_keepT <T, vecT, vec_size> (&M_row[n_off], shm, block_dim_m, last_m_dim, ld_m, false);

      for (int i2 = 0; i2 < iters_k; i2++)
      {
        const int k_off = i2 * block_dim_k; const T * A_k = &A_row[k_off * A_step_c], * B_k = &B_col[k_off * B_step_r];
        const int ld_1 = matrixCopy_keepT <T, vecT, vec_size> (A_k, shm_a, block_dim_k, last_m_dim, ld_a, a_T);
        const int ld_2 = matrixCopy_keepT <T, vecT, vec_size> (B_k, shm_b, block_dim_m, block_dim_k, ld_b, b_T);
        __syncthreads();

        DenseGemm <T, vecT, vec_size> (shm, shm_a, shm_b, last_m_dim, block_dim_m, block_dim_k, ld_0, ld_1, ld_2, a_T, b_T);
        __syncthreads();
      }

      if (last_k_dim > 0)
      {
        const int k_off = iters_k * block_dim_k; const T * A_k = &A_row[k_off * A_step_c], * B_k = &B_col[k_off * B_step_r];
        const int ld_1 = matrixCopy_keepT <T, vecT, vec_size> (A_k, shm_a, last_k_dim, last_m_dim, ld_a, a_T);
        const int ld_2 = matrixCopy_keepT <T, vecT, vec_size> (B_k, shm_b, block_dim_m, last_k_dim, ld_b, b_T);
        __syncthreads();

        DenseGemm <T, vecT, vec_size> (shm, shm_a, shm_b, last_m_dim, block_dim_m, last_k_dim, ld_0, ld_1, ld_2, a_T, b_T);
        __syncthreads();
      }

      matrixCopy <T, vecT, vec_size> (shm, &M_row[n_off], block_dim_m, last_m_dim, ld_0, ld_m);
      __syncthreads();
    }

    if (last_n_dim > 0)
    {
      const int n_off = iters_n * block_dim_m; const T * B_col = &B[n_off * B_step_c];
      const int ld_0 = matrixCopy_keepT <T, vecT, vec_size> (&M_row[n_off], shm, last_n_dim, block_dim_m, ld_m, false);

      for (int i2 = 0; i2 < iters_k; i2++)
      {
        const int k_off = i2 * block_dim_k; const T * A_k = &A_row[k_off * A_step_c], * B_k = &B_col[k_off * B_step_r];
        const int ld_1 = matrixCopy_keepT <T, vecT, vec_size> (A_k, shm_a, block_dim_k, last_m_dim, ld_a, a_T);
        const int ld_2 = matrixCopy_keepT <T, vecT, vec_size> (B_k, shm_b, last_n_dim, block_dim_k, ld_b, b_T);
        __syncthreads();

        DenseGemm <T, vecT, vec_size> (shm, shm_a, shm_b, last_m_dim, last_n_dim, block_dim_k, ld_0, ld_1, ld_2, a_T, b_T);
        __syncthreads();
      }

      if (last_k_dim > 0)
      {
        const int k_off = iters_k * block_dim_k; const T * A_k = &A_row[k_off * A_step_c], * B_k = &B_col[k_off * B_step_r];
        const int ld_1 = matrixCopy_keepT <T, vecT, vec_size> (A_k, shm_a, last_k_dim, last_m_dim, ld_a, a_T);
        const int ld_2 = matrixCopy_keepT <T, vecT, vec_size> (B_k, shm_b, last_n_dim, last_k_dim, ld_b, b_T);
        __syncthreads();

        DenseGemm <T, vecT, vec_size> (shm, shm_a, shm_b, last_m_dim, last_n_dim, last_k_dim, ld_0, ld_1, ld_2, a_T, b_T);
        __syncthreads();
      }

      matrixCopy <T, vecT, vec_size> (shm, &M_row[n_off], last_n_dim, last_m_dim, ld_0, ld_m);
      __syncthreads();
    }
  }

  if (alpha != 1.)
  {
    T thread_a[vec_size];

    for (int row = w_id; row < m; row += n_wp)
    {
      T * M_row = &M[row * ld_m];

      for (int col = l_id; col < iter_n; col += warpSize)
      {
        reinterpret_cast <vecT *> (thread_a)[0] = reinterpret_cast <vecT *> (M_row)[col];

        #pragma unroll
        for (int i = 0; i < vec_size; i++)
        { thread_a[i] *= alpha; }

        reinterpret_cast <vecT *> (M_row)[col] = reinterpret_cast <vecT *> (thread_a)[0];
      }
    }

    if (b_last_n)
    for (int row = w_id; row < m; row += n_wp)
    {
      T * M_row = &M[row * ld_m + last_n_start];

      for (int col = l_id; col < last_n; col += warpSize)
      { M_row[col] *= alpha; }
    }
  }
  __syncthreads();

}

template <class T, class vecT, int vec_size, int block_dim_m, int block_dim_k>
/* L is ny_l x nx_l lower triangular and unit diagonal, B is ny_l by nx_b, solves L x X = B, overwrites X in B. */
__device__ __forceinline__ void blockDenseTrsmL (T * __restrict__ B, const T * __restrict__ L, const int nx_b, const int ny_b, const int nx_l, const int ld_b, const int ld_l, T * __restrict__ shm)
{
  const int l_step = block_dim_m * ld_l + block_dim_m, b_step = block_dim_m * ld_b; int remain_nx = nx_l, remain_ny = ny_b;
  const T * L_diag = L, * L_left = &L[l_step - block_dim_m];
  T * B_top = B, * B_next = &B[b_step];

  while (remain_nx > block_dim_m && remain_ny > block_dim_m)
  {
    remain_nx -= block_dim_m;
    remain_ny -= block_dim_m;

    const int ld_0 = matrixCopy_keepT <T, vecT, vec_size> (L_diag, shm, block_dim_m, block_dim_m, ld_l, false);
    __syncthreads();

    DenseTrsmL <T, vecT, vec_size> (B_top, shm, nx_b, block_dim_m, block_dim_m, ld_b, ld_0);

    blockDenseGemm <T, vecT, vec_size, block_dim_m, block_dim_k> (-1., 1., B_next, L_left, B_top, remain_ny, nx_b, block_dim_m, ld_b, ld_l, ld_b, false, false, shm);

    L_diag = &L_diag[l_step];
    L_left = &L_left[l_step];
    B_top = B_next;
    B_next = &B_next[b_step];
  }

  if (remain_nx <= block_dim_m && remain_ny <= block_dim_m)
  {
    const int ld_0 = matrixCopy_keepT <T, vecT, vec_size> (L_diag, shm, remain_nx, remain_ny, ld_l, false);
    __syncthreads();

    DenseTrsmL <T, vecT, vec_size> (B_top, shm, nx_b, remain_ny, remain_nx, ld_b, ld_0);
  }
  else if (remain_nx <= block_dim_m)
  {
    const int ld_0 = matrixCopy_keepT <T, vecT, vec_size> (L_diag, shm, remain_nx, block_dim_m, ld_l, false);
    __syncthreads();

    DenseTrsmL <T, vecT, vec_size> (B_top, shm, nx_b, block_dim_m, remain_nx, ld_b, ld_0);

    blockDenseGemm <T, vecT, vec_size, block_dim_m, block_dim_k> (-1., 1., B_next, L_left, B_top, remain_ny - block_dim_m, nx_b, remain_nx, ld_b, ld_l, ld_b, false, false, shm);
  }
  else if (remain_ny <= block_dim_m)
  {
    const int ld_0 = matrixCopy_keepT <T, vecT, vec_size> (L_diag, shm, block_dim_m, remain_ny, ld_l, false);
    __syncthreads();

    DenseTrsmL <T, vecT, vec_size> (B_top, shm, nx_b, remain_ny, block_dim_m, ld_b, ld_0);
  }

}

template <class T, class vecT, int vec_size, int block_dim_m, int block_dim_k>
/* U is ny_u x nx_u upper triangular and not unit diagonal, B is ny_b by nx_u, solves X x U = B, overwrites X in B. */
__device__ __forceinline__ void blockDenseTrsmR (T * __restrict__ B, const T * __restrict__ U, const int nx_b, const int ny_b, const int ny_u, const int ld_b, const int ld_u, T * __restrict__ shm)
{
  const int u_step = block_dim_m * ld_u + block_dim_m, b_step = block_dim_m; int remain_nx = nx_b, remain_ny = ny_u;
  const T * U_diag = U, * U_top = &U[block_dim_m];
  T * B_left = B, * B_next = &B[b_step];

  while (remain_nx > block_dim_m && remain_ny > block_dim_m)
  {
    remain_nx -= block_dim_m;
    remain_ny -= block_dim_m;

    const int ld_0 = matrixCopy_keepT <T, vecT, vec_size> (U_diag, shm, block_dim_m, block_dim_m, ld_u, false);
    __syncthreads();

    DenseTrsmR <T, vecT, vec_size> (B_left, shm, block_dim_m, ny_b, block_dim_m, ld_b, ld_0);

    blockDenseGemm <T, vecT, vec_size, block_dim_m, block_dim_k> (-1., 1., B_next, B_left, U_top, ny_b, remain_nx, block_dim_m, ld_b, ld_b, ld_u, false, false, shm);

    U_diag = &U_diag[u_step];
    U_top = &U_top[u_step];
    B_left = B_next;
    B_next = &B_next[b_step];
  }

  if (remain_nx <= block_dim_m && remain_ny <= block_dim_m)
  {
    const int ld_0 = matrixCopy_keepT <T, vecT, vec_size> (U_diag, shm, remain_nx, remain_ny, ld_u, false);
    __syncthreads();

    DenseTrsmR <T, vecT, vec_size> (B_left, shm, remain_nx, ny_b, remain_ny, ld_b, ld_0);
  }
  else if (remain_nx <= block_dim_m)
  {
    const int ld_0 = matrixCopy_keepT <T, vecT, vec_size> (U_diag, shm, remain_nx, block_dim_m, ld_u, false);
    __syncthreads();

    DenseTrsmR <T, vecT, vec_size> (B_left, shm, remain_nx, ny_b, block_dim_m, ld_b, ld_0);
  }
  else if (remain_ny <= block_dim_m)
  {
    const int ld_0 = matrixCopy_keepT <T, vecT, vec_size> (U_diag, shm, block_dim_m, remain_ny, ld_u, false);
    __syncthreads();

    DenseTrsmR <T, vecT, vec_size> (B_left, shm, block_dim_m, ny_b, remain_ny, ld_b, ld_0);

    blockDenseGemm <T, vecT, vec_size, block_dim_m, block_dim_k> (-1., 1., B_next, B_left, U_top, ny_b, remain_nx - block_dim_m, remain_ny, ld_b, ld_b, ld_u, false, false, shm);
  }

}

template <class T, class vecT, int vec_size, int block_dim_m, int block_dim_k>
/* U is ny_u x nx_u upper triangular and not unit diagonal, B is ny_b by nx_u, solves X x U = B, overwrites X in B. */
__device__ __forceinline__ void blockDenseTrsmR_transposeB (T * __restrict__ B, const T * __restrict__ U, const int nx_b, const int ny_b, const int ny_u, const int ld_b, const int ld_u, T * __restrict__ shm)
{
  const int u_step = block_dim_m * ld_u + block_dim_m, b_step = block_dim_m * ld_b; int remain_nx = nx_b, remain_ny = ny_u;
  const T * U_diag = U, * U_top = &U[block_dim_m];
  T * B_left = B, * B_next = &B[b_step];

  while (remain_nx > block_dim_m && remain_ny > block_dim_m)
  {
    remain_nx -= block_dim_m;
    remain_ny -= block_dim_m;

    const int ld_0 = matrixCopy_keepT <T, vecT, vec_size> (U_diag, shm, block_dim_m, block_dim_m, ld_u, false);
    __syncthreads();

    DenseTrsmR_transposeB <T, vecT, vec_size> (B_left, shm, block_dim_m, ny_b, block_dim_m, ld_b, ld_0);

    blockDenseGemm <T, vecT, vec_size, block_dim_m, block_dim_k> (-1., 1., B_next, U_top, B_left, remain_nx, ny_b, block_dim_m, ld_b, ld_u, ld_b, true, false, shm);

    U_diag = &U_diag[u_step];
    U_top = &U_top[u_step];
    B_left = B_next;
    B_next = &B_next[b_step];
  }

  if (remain_nx <= block_dim_m && remain_ny <= block_dim_m)
  {
    const int ld_0 = matrixCopy_keepT <T, vecT, vec_size> (U_diag, shm, remain_nx, remain_ny, ld_u, false);
    __syncthreads();

    DenseTrsmR_transposeB <T, vecT, vec_size> (B_left, shm, remain_nx, ny_b, remain_ny, ld_b, ld_0);
  }
  else if (remain_nx <= block_dim_m)
  {
    const int ld_0 = matrixCopy_keepT <T, vecT, vec_size> (U_diag, shm, remain_nx, block_dim_m, ld_u, false);
    __syncthreads();

    DenseTrsmR_transposeB <T, vecT, vec_size> (B_left, shm, remain_nx, ny_b, block_dim_m, ld_b, ld_0);
  }
  else if (remain_ny <= block_dim_m)
  {
    const int ld_0 = matrixCopy_keepT <T, vecT, vec_size> (U_diag, shm, block_dim_m, remain_ny, ld_u, false);
    __syncthreads();

    DenseTrsmR_transposeB <T, vecT, vec_size> (B_left, shm, block_dim_m, ny_b, remain_ny, ld_b, ld_0);

    blockDenseGemm <T, vecT, vec_size, block_dim_m, block_dim_k> (-1., 1., B_next, U_top, B_left, remain_nx - block_dim_m, ny_b, remain_ny, ld_b, ld_u, ld_b, true, false, shm);
  }

}

template <class T, class vecT, int vec_size, int block_dim_m, int block_dim_k>
/* LU decomposition of matrix of ny by nx, utilizes L1 cache. */
__device__ __forceinline__ void blockDenseGetrf (T * __restrict__ M, const int nx, const int ny, const int ld, T * __restrict__ shm)
{
  const int iter_step = block_dim_m * ld + block_dim_m; int remain_nx = nx, remain_ny = ny;
  T * M_diag = M, * M_top = &M[block_dim_m], * M_left = &M[iter_step - block_dim_m], * M_next = &M[iter_step];

  while (remain_nx > block_dim_m && remain_ny > block_dim_m)
  {
    remain_nx -= block_dim_m;
    remain_ny -= block_dim_m;

    const int ld_0 = matrixCopy_keepT <T, vecT, vec_size> (M_diag, shm, block_dim_m, block_dim_m, ld, false);
    __syncthreads();

    DenseGetrf <T, vecT, vec_size> (shm, block_dim_m, block_dim_m, ld_0);

    matrixCopy <T, vecT, vec_size> (shm, M_diag, block_dim_m, block_dim_m, ld_0, ld);

    DenseTrsmL <T, vecT, vec_size> (M_top, shm, remain_nx, block_dim_m, block_dim_m, ld, ld_0);

    DenseTrsmR <T, vecT, vec_size> (M_left, shm, block_dim_m, remain_ny, block_dim_m, ld, ld_0);

    blockDenseGemm <T, vecT, vec_size, block_dim_m, block_dim_k> (-1., 1., M_next, M_left, M_top, remain_ny, remain_nx, block_dim_m, ld, ld, ld, false, false, shm);

    M_diag = M_next;
    M_left = &M_left[iter_step];
    M_top = &M_top[iter_step];
    M_next = &M_next[iter_step];
  }

  if (remain_nx <= block_dim_m && remain_ny <= block_dim_m)
  {
    const int ld_0 = matrixCopy_keepT <T, vecT, vec_size> (M_diag, shm, remain_nx, remain_ny, ld, false);
    __syncthreads();

    DenseGetrf <T, vecT, vec_size> (shm, remain_nx, remain_ny, ld_0);

    matrixCopy <T, vecT, vec_size> (shm, M_diag, remain_nx, remain_ny, ld_0, ld);
    __syncthreads();
  }
  else if (remain_ny <= block_dim_m)
  {
    const int ld_0 = matrixCopy_keepT <T, vecT, vec_size> (M_diag, shm, block_dim_m, remain_ny, ld, false);
    __syncthreads();

    DenseGetrf <T, vecT, vec_size> (shm, block_dim_m, remain_ny, ld_0);

    matrixCopy <T, vecT, vec_size> (shm, M_diag, block_dim_m, remain_ny, ld_0, ld);

    DenseTrsmL <T, vecT, vec_size> (M_top, shm, remain_nx - block_dim_m, remain_ny, block_dim_m, ld, ld_0);
  }
  else if (remain_nx <= block_dim_m)
  {
    const int ld_0 = matrixCopy_keepT <T, vecT, vec_size> (M_diag, shm, remain_nx, block_dim_m, ld, false);
    __syncthreads();

    DenseGetrf <T, vecT, vec_size> (shm, remain_nx, block_dim_m, ld_0);

    matrixCopy <T, vecT, vec_size> (shm, M_diag, remain_nx, block_dim_m, ld_0, ld);

    DenseTrsmR <T, vecT, vec_size> (M_left, shm, remain_nx, remain_ny - block_dim_m, block_dim_m, ld, ld_0);
  }

}


template <class T, class vecT, int vec_size, int block_dim_m, int block_dim_k>
/* General Matrix multiplication with 3 matrices. M (m by n) = alpha * A (m by k) * B (k by l) * C (l by n) + beta * old_M. */
__device__ __forceinline__ void blockDenseGemm_3x (const T alpha, const T beta, T * __restrict__ M, const T * __restrict__ A, const T * __restrict__ B, 
  const T * __restrict__ C, const int m, const int n, const int k, const int l, const int ld_m, const int ld_a, const int ld_b, const int ld_c, 
  const bool a_T, const bool b_T, const bool c_T, const int control, const int t_size1, T * __restrict__ shm)
{
  const int t_id = thread_rank();

  T * t1, ** t1_addr = (T **) &shm[0];

  if (t_id == 0)
  { * t1_addr = new T[t_size1]; }
  __syncthreads();

  t1 = * t1_addr;
  __syncthreads();

  if (control) // (A x B) x C
  {
    blockDenseGemm <T, vecT, vec_size, block_dim_m, block_dim_k> (1., 0., t1, A, B, m, l, k, l, ld_a, ld_b, a_T, b_T, shm);
    blockDenseGemm <T, vecT, vec_size, block_dim_m, block_dim_k> (alpha, beta, M, t1, C, m, n, l, ld_m, l, ld_c, false, c_T, shm);
  }
  else // A x (B x C)
  {
    blockDenseGemm <T, vecT, vec_size, block_dim_m, block_dim_k> (1., 0., t1, B, C, k, n, l, n, ld_b, ld_c, b_T, c_T, shm);
    blockDenseGemm <T, vecT, vec_size, block_dim_m, block_dim_k> (alpha, beta, M, A, t1, m, n, k, ld_m, ld_a, n, a_T, false, shm);
  }


  if (t_id == 0)
  { delete[] t1; }
  __syncthreads();

}

template <class T, class vecT, int vec_size, int block_dim_m, int block_dim_k>
/* General Matrix multiplication with 4 matrices. M (m by n) = alpha * A (m by k) * B (k by l) * C (l by o) * D (o by n) + beta * old_M. */
__device__ void blockDenseGemm_4x (const T alpha, const T beta, T * __restrict__ M, const T * __restrict__ A, const T * __restrict__ B, const T * __restrict__ C, 
  const T * __restrict__ D, const int m, const int n, const int k, const int l, const int o, const int ld_m, const int ld_a, const int ld_b, const int ld_c, 
  const int ld_d, const bool a_T, const bool b_T, const bool c_T, const bool d_T, const int control, const int t_size1, const int t_size2, T * __restrict__ shm)
{
  const int t_id = thread_rank();

  T * t1, ** t1_addr = (T **) &shm[0], * t2, ** t2_addr = (T **) & shm[1];

  if (t_id == 0)
  { * t1_addr = new T[t_size1]; * t2_addr = new T[t_size2]; }
  __syncthreads();

  t1 = * t1_addr;
  t2 = * t2_addr;
  __syncthreads();

  switch (control) 
  {
  case 0: // ((A x B) x C) x D, t1 m * l, t2 m * o
  {
    blockDenseGemm <T, vecT, vec_size, block_dim_m, block_dim_k> (1., 0., t1, A, B, m, l, k, l, ld_a, ld_b, a_T, b_T, shm);
    blockDenseGemm <T, vecT, vec_size, block_dim_m, block_dim_k> (1., 0., t2, t1, C, m, o, l, o, l, ld_c, false, c_T, shm);
    blockDenseGemm <T, vecT, vec_size, block_dim_m, block_dim_k> (alpha, beta, M, t2, D, m, n, o, ld_m, o, ld_d, false, d_T, shm);
    break;
  }
  case 1: // (A x (B x C)) x D, t1 k * o, t2 m * o
  {
    blockDenseGemm <T, vecT, vec_size, block_dim_m, block_dim_k> (1., 0., t1, B, C, k, o, l, o, ld_b, ld_c, b_T, c_T, shm);
    blockDenseGemm <T, vecT, vec_size, block_dim_m, block_dim_k> (1., 0., t2, A, t1, m, o, k, o, ld_a, o, a_T, false, shm);
    blockDenseGemm <T, vecT, vec_size, block_dim_m, block_dim_k> (alpha, beta, M, t2, D, m, n, o, ld_m, o, ld_d, false, d_T, shm);
    break;
  }
  case 2: // A x ((B x C) x D), t1 k * o, t2 k * n
  {
    blockDenseGemm <T, vecT, vec_size, block_dim_m, block_dim_k> (1., 0., t1, B, C, k, o, l, o, ld_b, ld_c, b_T, c_T, shm);
    blockDenseGemm <T, vecT, vec_size, block_dim_m, block_dim_k> (1., 0., t2, t1, D, k, n, o, n, o, ld_d, false, d_T, shm);
    blockDenseGemm <T, vecT, vec_size, block_dim_m, block_dim_k> (alpha, beta, M, A, t2, m, n, k, ld_m, ld_a, n, a_T, false, shm);
    break;
  }
  case 3: // A x (B x (C x D)), t1 l * n, t2 k * n
  {
    blockDenseGemm <T, vecT, vec_size, block_dim_m, block_dim_k> (1., 0., t1, C, D, l, n, o, n, ld_c, ld_d, c_T, d_T, shm);
    blockDenseGemm <T, vecT, vec_size, block_dim_m, block_dim_k> (1., 0., t2, B, t1, k, n, l, n, ld_b, n, b_T, false, shm);
    blockDenseGemm <T, vecT, vec_size, block_dim_m, block_dim_k> (alpha, beta, M, A, t2, m, n, k, ld_m, ld_a, n, a_T, false, shm);
    break;
  }
  case 4: // (A x B) x (C x D), t1 m * l, t2 l * n
  {
    blockDenseGemm <T, vecT, vec_size, block_dim_m, block_dim_k> (1., 0., t1, A, B, m, l, k, l, ld_a, ld_b, a_T, b_T, shm);
    blockDenseGemm <T, vecT, vec_size, block_dim_m, block_dim_k> (1., 0., t2, C, D, l, n, o, n, ld_c, ld_d, c_T, d_T, shm);
    blockDenseGemm <T, vecT, vec_size, block_dim_m, block_dim_k> (alpha, beta, M, t1, t2, m, n, l, ld_m, l, n, false, false, shm);
    break;
  }
  default:
  { break; }
  }

  if (t_id == 0)
  { delete[] t1; delete[] t2; }
  __syncthreads();

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




#endif