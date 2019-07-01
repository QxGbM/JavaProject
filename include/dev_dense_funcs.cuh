
#ifndef _DEV_DENSE_FUNCS_CUH
#define _DEV_DENSE_FUNCS_CUH

#include <pspl.cuh>

template <class T, int step_size>
/* A convinient call to copy from shared memory to global or vice versa. Reading "from" in row major. */
__device__ __forceinline__ void matrixCopy (const T * __restrict__ from, T * __restrict__ to, const int nx_to, const int ny_to, const int ld_from, const int ld_to, const bool transpose)
{
  const int w_id = warp_rank(), l_id = lane_rank(), n_wp = num_warps(), iter_step = step_size * n_wp;

  if (transpose)
  {
    for (int row_start = w_id * step_size; row_start < nx_to; row_start += iter_step)
    {
      #pragma unroll step_size
      for (int row = 0; row < step_size; row ++)
      {
        if (row < nx_to)
        {
          const int row_at = row + row_start;
          const T * from_row = &from[row_at * ld_from];
          for (int col = l_id; col < ny_to; col += warpSize)
          { to[col * ld_to + row_at] = from_row[col]; }
        }
      }
    }

  }
  else
  {
    for (int row_start = w_id * step_size; row_start < ny_to; row_start += iter_step)
    {
      #pragma unroll step_size
      for (int row = 0; row < step_size; row ++)
      {
        if (row < ny_to)
        {
          const int row_at = row + row_start;
          T * to_row = &to[row_at * ld_to];
          const T * from_row = &from[row_at * ld_from];
          for (int col = l_id; col < nx_to; col += warpSize)
          { to_row[col] = from_row[col]; }
        }
      }
    }

  }
}

template <class T, int step_size>
/* A convinient call to copy from shared memory to global or vice versa. Reading "from" in row major. */
__device__ __forceinline__ int matrixCopy_keepT (const T * __restrict__ from, T * __restrict__ to, const int nx_to, const int ny_to, const int ld_from, const bool transpose)
{
  const int w_id = warp_rank(), l_id = lane_rank(), n_wp = num_warps(), iter_step = step_size * n_wp;

  if (transpose)
  {
    const int ld_to = ny_to;

    for (int row_start = w_id * step_size; row_start < nx_to; row_start += iter_step)
    {
      #pragma unroll step_size
      for (int row = 0; row < step_size; row ++)
      {
        if (row < nx_to)
        {
          const int row_at = row + row_start;
          T * to_row = &to[row_at * ld_to];
          const T * from_row = &from[row_at * ld_from];
          for (int col = l_id; col < ny_to; col += warpSize)
          { to_row[col] = from_row[col]; }
        }
      }
    }

    return ld_to;
  }
  else
  {
    const int ld_to = nx_to;

    for (int row_start = w_id * step_size; row_start < ny_to; row_start += iter_step)
    {
      #pragma unroll step_size
      for (int row = 0; row < step_size; row ++)
      {
        if (row < ny_to)
        {
          const int row_at = row + row_start;
          T * to_row = &to[row_at * ld_to];
          const T * from_row = &from[row_at * ld_from];
          for (int col = l_id; col < nx_to; col += warpSize)
          { to_row[col] = from_row[col]; }
        }
      }
    }

    return ld_to;
  }

}


template <class T> 
/* LU decomposition of matrix of ny by nx. */
__device__ __forceinline__ void DenseGetrf (T * M, const int nx, const int ny, const int ld)
{
  const int w_id = warp_rank(), l_id = lane_rank(), n_wp = num_warps(), min_n = nx > ny ? ny : nx;
  
  for (int i = 0; i < min_n; i ++)
  {
    for (int row = w_id + i + 1; row < ny; row += n_wp)
    {
      T left, * M_top = &M[i * ld], * M_row = &M[row * ld];

      if (l_id == 0)
      { left = M_row[i] / M_top[i]; M_row[i] = left; }
      __syncwarp();

      left = __shfl_sync (0xffffffff, left, 0, warpSize);

      for (int col = l_id + i + 1; col < nx; col += warpSize)
      { M_row[col] -= left * M_top[col]; }
    }
    __syncthreads();
  }

}

template <class T>
/* L is ny_l x nx_l lower triangular and unit diagonal, B is ny_l by nx_b, solves L x X = B, overwrites X in B. */
__device__ __forceinline__ void DenseTrsmL (T * __restrict__ B, const T * __restrict__ L, const int nx_b, const int ny_b, const int nx_l, const int ld_b, const int ld_l)
{
  const int w_id = warp_rank(), l_id = lane_rank(), n_wp = num_warps(), min_n = nx_l > ny_b ? ny_b : nx_l;

  for (int i = 0; i < min_n; i ++)
  {
    for (int row = w_id + i + 1; row < ny_b; row += n_wp)
    {
      T left = L[row * ld_l + i], * B_top = &B[i * ld_b], * B_row = &B[row * ld_b];

      for (int col = l_id; col < nx_b; col += warpSize)
      { B_row[col] -= left * B_top[col]; }
    }
    __syncthreads();
  }
}

template <class T>
/* U is ny_u x nx_u upper triangular and not unit diagonal, B is ny_b by nx_u, solves X x U = B, overwrites X in B. */
__device__ __forceinline__ void DenseTrsmR (T * __restrict__ B, const T * __restrict__ U, const int nx_b, const int ny_b, const int ny_u, const int ld_b, const int ld_u)
{
  const int w_id = warp_rank(), l_id = lane_rank(), n_wp = num_warps(), min_n = nx_b > ny_u ? ny_u : nx_b;

  for (int i = 0; i < min_n; i ++)
  {
    for (int row = w_id; row < ny_b; row += n_wp)
    {
      T left, * B_row = &B[row * ld_b];
      const T * U_top = &U[i * ld_u];

      if (l_id == 0)
      { left = B_row[i] / U_top[i]; B_row[i] = left; }
      __syncwarp();

      left = __shfl_sync (0xffffffff, left, 0, warpSize);

      for (int col = l_id + i + 1; col < nx_b; col += warpSize)
      { B_row[col] -= left * U_top[col]; }
    }
    __syncthreads();
  }
}

template <class T>
/* U is ny_u x nx_u upper triangular and not unit diagonal, B is ny_b by nx_u, solves X x U = B, overwrites X in B. */
__device__ __forceinline__ void DenseTrsmR_transposeB (T * __restrict__ B, const T * __restrict__ U, const int nx_b, const int ny_b, const int ny_u, const int ld_b, const int ld_u)
{
  const int w_id = warp_rank(), l_id = lane_rank(), n_wp = num_warps(), min_n = nx_b > ny_u ? ny_u : nx_b;

  for (int i = 0; i < min_n; i ++)
  {
    T * B_top = &B[i * ld_b];
    const T * U_top = &U[i * ld_u];

    if (w_id == 0)
    {
      for (int col = l_id; col < ny_b; col += warpSize)
      { B_top[col] /= U_top[i]; }
    }
    __syncthreads();

    for (int row = w_id + i + 1; row < nx_b; row += n_wp)
    {
      T left = U_top[row], * B_row = &B[row * ld_b];

      for (int col = l_id; col < ny_b; col += warpSize)
      { B_row[col] -= left * B_top[col]; }
    }
    __syncthreads();
  }
}


template <class T>
/* General Matrix multiplication. M (m by n) = alpha * A (m by k) * B (k by n) + beta * old_M. */
__device__ __forceinline__ void DenseGemm (const T alpha, const T beta, T * __restrict__ M, const T * __restrict__ A, const T * __restrict__ B,
  const int m, const int n, const int k, const int ld_m, const int ld_a, const int ld_b, const bool a_T, const bool b_T)
{
  const int w_id = warp_rank(), l_id = lane_rank(), n_wp = num_warps();

  if (beta == 0.)
  for (int row = w_id; row < m; row += n_wp)
  {
    T * M_row = &M[row * ld_m];
    for (int col = l_id; col < n; col += warpSize)
    { M_row[col] = 0.; }
  }
  else if (beta != 1.)
  for (int row = w_id; row < m; row += n_wp)
  {
    T * M_row = &M[row * ld_m];
    for (int col = l_id; col < n; col += warpSize)
    { M_row[col] *= beta; }
  }
  __syncthreads();

  if (alpha != 0.)
  {
    const int A_step = a_T ? ld_a : 1, B_step = b_T ? 1 : ld_b;

    for (int row = w_id; row < m; row += n_wp) 
    {
      T * M_row = &M[row * ld_m];
      const T * A_start = a_T ? &A[row] : &A[row * ld_a];

      for (int col = l_id; col < n; col += warpSize)
      {
        T accum = 0.;
        const T * B_start = b_T ? &B[col * ld_b] : &B[col];

        for (int i = 0; i < k; i++)
        { accum += A_start[i * A_step] * B_start[i * B_step]; }

        M_row[col] += alpha * accum;
      }
    }
  }
  __syncthreads();

}

template <class T, int block_dim_m, int block_dim_k, int step_size>
/* General Matrix multiplication. M (m by n) = alpha * A (m by k) * B (k by n) + beta * old_M. */
__device__ __forceinline__ void blockDenseGemm (const T alpha, const T beta, T * __restrict__ M, const T * __restrict__ A, const T * __restrict__ B,
  const int m, const int n, const int k, const int ld_m, const int ld_a, const int ld_b, const bool a_T, const bool b_T, T * __restrict__ shm)
{
  const int w_id = warp_rank(), l_id = lane_rank(), n_wp = num_warps();

  if (beta == 0.)
  for (int row = w_id; row < m; row += n_wp)
  {
    T * M_row = &M[row * ld_m];
    for (int col = l_id; col < n; col += warpSize)
    { M_row[col] = 0.; }
  }
  else if (beta != 1.)
  for (int row = w_id; row < m; row += n_wp)
  {
    T * M_row = &M[row * ld_m];
    for (int col = l_id; col < n; col += warpSize)
    { M_row[col] *= beta; }
  }
  __syncthreads();

  T * shm_2 = &shm[block_dim_m * block_dim_k];
  
  for (int i = 0; i < n; i += block_dim_m)
  {
    const int n_remain = n - i, n_block = n_remain > block_dim_m ? block_dim_m : n_remain;

    for (int j = 0; j < k; j += block_dim_k)
    {
      const int k_remain = k - j, k_block = k_remain > block_dim_k ? block_dim_k : k_remain;
      const int B_offset = b_T ? i * ld_b + j : j * ld_b + i;

      const int ld_2 = matrixCopy_keepT <T, step_size> (&B[B_offset], shm_2, n_block, k_block, ld_b, b_T);
      __syncthreads();

      for (int l = 0; l < m; l += block_dim_m)
      {
        const int m_remain = m - l, m_block = m_remain > block_dim_m ? block_dim_m : m_remain;
        const int A_offset = a_T ? j * ld_a + l : l * ld_a + j;

        const int ld_1 = matrixCopy_keepT <T, step_size> (&A[A_offset], shm, k_block, m_block, ld_a, a_T);
        __syncthreads();

        DenseGemm <T> (alpha, 1., &M[l * ld_m + i], shm, shm_2, m_block, n_block, k_block, ld_m, ld_1, ld_2, a_T, b_T);

      }

    }

  }

}

template <class T, int block_dim_m, int block_dim_k, int step_size> 
/* LU decomposition of matrix of ny by nx, utilizes L1 cache. */
__device__ __forceinline__ void blockDenseGetrf (T * __restrict__ M, const int nx, const int ny, const int ld, T * __restrict__ shm)
{
  const int min_n = nx > ny ? ny : nx;

  for (int i = 0; i < min_n; i += block_dim_m)
  {
    int diag_nx, diag_ny, remain_nx, remain_ny;

    if (nx - i >= block_dim_m)
    { diag_nx = block_dim_m; remain_nx = nx - i - block_dim_m; }
    else
    { diag_nx = nx - i; remain_nx = 0; }

    if (ny - i >= block_dim_m)
    { diag_ny = block_dim_m; remain_ny = ny - i - block_dim_m; }
    else
    { diag_ny = ny - i; remain_ny = 0; }

    T * M_diag = &M[i * ld + i], * M_top = &M_diag[diag_nx], * M_left = &M_diag[diag_ny * ld], * M_next = &M_left[diag_nx];

    matrixCopy_keepT <T, step_size> (M_diag, shm, diag_nx, diag_ny, ld, false);
    __syncthreads();

    DenseGetrf <T> (shm, diag_nx, diag_ny, diag_nx);

    matrixCopy <T, step_size> (shm, M_diag, diag_nx, diag_ny, diag_nx, ld, false);

    bool solve_row = remain_nx > 0, solve_col = remain_ny > 0;

    if (solve_row)
    { DenseTrsmL <T> (M_top, shm, remain_nx, diag_ny, diag_nx, ld, diag_nx); }

    if (solve_col)
    { DenseTrsmR <T> (M_left, shm, diag_nx, remain_ny, diag_ny, ld, diag_nx); }

    __syncthreads();

    if (solve_row && solve_col)
    { blockDenseGemm <T, block_dim_m, block_dim_k, step_size> (-1., 1., M_next, M_left, M_top, remain_ny, remain_nx, block_dim_m, ld, ld, ld, false, false, shm); }
  }

}


template <class T, int block_dim_m, int block_dim_k, int step_size>
/* L is ny_l x nx_l lower triangular and unit diagonal, B is ny_l by nx_b, solves L x X = B, overwrites X in B. */
__device__ __forceinline__ void blockDenseTrsmL (T * __restrict__ B, const T * __restrict__ L, const int nx_b, const int ny_b, const int nx_l, const int ld_b, const int ld_l, T * __restrict__ shm)
{
  const int min_n = nx_l > ny_b ? ny_b : nx_l;

  for (int i = 0; i < min_n; i += block_dim_m)
  {
    int diag_ny, remain_ny;

    if (ny_b - i >= block_dim_m)
    { diag_ny = block_dim_m; remain_ny = ny_b - i - block_dim_m; }
    else
    { diag_ny = ny_b - i; remain_ny = 0; }

    const T * L_diag = &L[i * ld_l + i], * L_left = &L_diag[diag_ny * ld_l];
    T * B_top = &B[i * ld_b], * B_next = &B_top[diag_ny * ld_b];

    matrixCopy_keepT <T, step_size> (L_diag, shm, diag_ny, diag_ny, ld_l, false);
    __syncthreads();

    DenseTrsmL <T> (B_top, shm, nx_b, diag_ny, diag_ny, ld_b, diag_ny);

    if (remain_ny > 0)
    { blockDenseGemm <T, block_dim_m, block_dim_k, step_size> (-1., 1., B_next, L_left, B_top, remain_ny, nx_b, block_dim_m, ld_b, ld_l, ld_b, false, false, shm); }

  }
}

template <class T, int block_dim_m, int block_dim_k, int step_size>
/* U is ny_u x nx_u upper triangular and not unit diagonal, B is ny_b by nx_u, solves X x U = B, overwrites X in B. */
__device__ __forceinline__ void blockDenseTrsmR (T * __restrict__ B, const T * __restrict__ U, const int nx_b, const int ny_b, const int ny_u, const int ld_b, const int ld_u, T * __restrict__ shm)
{
  DenseTrsmR <T> (B, U, nx_b, ny_b, ny_u, ld_b, ld_u);
}

template <class T, int block_dim_m, int block_dim_k, int step_size>
/* U is ny_u x nx_u upper triangular and not unit diagonal, B is ny_b by nx_u, solves X x U = B, overwrites X in B. */
__device__ __forceinline__ void blockDenseTrsmR_transposeB (T * __restrict__ B, const T * __restrict__ U, const int nx_b, const int ny_b, const int ny_u, const int ld_b, const int ld_u, T * __restrict__ shm)
{
  DenseTrsmR_transposeB <T> (B, U, nx_b, ny_b, ny_u, ld_b, ld_u);

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
      matrixCopy <T, 1> (&A[a_T ? i0 * ld_a: i0], shm, k_, m, ld_a, k_, a_T);
      matrixCopy <T, 1> (&B[b_T ? i0 : i0 * ld_b], shm_B, k_, n, ld_b, k_, !b_T);
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
      matrixCopy <T, 1> (&A[a_T ? i0 * ld_a : i0], shm, k_, m, ld_a, k_, a_T);
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
      matrixCopy <T, 1> (&C[c_T ? i0 : i0 * ld_c], shm_C, l_, n, ld_c, l_, !c_T);
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
      matrixCopy <T, 1> (&D[d_T ? i0 : i0 * ld_d], shm_D, o_, n, ld_d, o_, !d_T);
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
      matrixCopy <T, 1> (&A[a_T ? i0 * ld_a : i0], shm, k_, m, ld_a, k_, a_T);
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