
#ifndef _DEV_DENSE_FUNCS_CUH
#define _DEV_DENSE_FUNCS_CUH

#include <pspl.cuh>

/* Scalar of a matrix of ny by nx. */
template <class T> __device__ void blockDenseScalar (const T scale, T *matrix, const int nx, const int ny, const int ld)
{
  const int thread_id = thread_rank();
  const int block_size = block_dim();
  for (int i = thread_id; i < nx * ny; i += block_size)
  { 
    const int row = i / nx, col = i - row * nx;
    matrix[row * ld + col] = (scale == 0) ? 0 : matrix[row * ld + col] * scale;
  }
  __syncthreads();
}

/* A has dimension m * k, B has dimension k * n, matrix has dimension m * n. matrix = alpha * A * B + beta * old_matrix. */
template <class T> __device__ void blockDenseGemm (const T alpha, const T beta, T *matrix, const T *a, const T *b, const int m, const int n, const int k, 
  const int ld_m, const int ld_a, const int ld_b)
{
  const int thread_id = thread_rank();
  const int block_size = block_dim();
  for (int i = thread_id; i < m * n; i += block_size)
  { 
    const int row = i / n, col = i - row * n;
    const T old = (beta == 0) ? 0 : beta * matrix[row * ld_m + col];
    T accum = 0;
    if (alpha != 0)
    {
      for (int j = 0; j < k; j++)
      { accum += a[row * ld_a + j] * b[j * ld_b + col]; }
      accum *= alpha;
    }
    matrix[row * ld_m + col] = old + accum;
  }
  __syncthreads();
}

/* Pivoted LU decomposition of matrix of ny by nx. */
template <class T> __device__ void blockDenseGetrf (T *matrix, const int nx, const int ny, const int ld, int *pivot = nullptr)
{
  if (pivot != nullptr) { resetPivot(pivot, ny); }

  for (int i = 0; i < nx && i < ny; i++)
  {
    if (pivot != nullptr)
    {
      const int target = i + blockAllFindRowPivot <T> (&matrix[i * ld + i], ny - i, ld);

      if (target != i)
      {
        blockSwapNSeqElements <T> (&matrix[target * ld], &matrix[i * ld], nx);
        if (thread_rank() == 0) { int t = pivot[target]; pivot[target] = pivot[i]; pivot[i] = t; }
        __syncthreads();
      }
    }

    blockDenseScalar <T> (1.0 / matrix[i * ld + i], &matrix[(i + 1) * ld + i], 1, ny - (i + 1), ld);

    blockDenseGemm <T> (-1.0, 1.0, &matrix[(i + 1) * ld + (i + 1)], &matrix[(i + 1) * ld + i], &matrix[i * ld + (i + 1)],
      ny - (i + 1), nx - (i + 1), 1, ld, ld, ld);
  }
}

/* L is ny_l x nx_l lower triangular and unit diagonal, B is ny_l by nx_b, solves L x X = B, overwrites X in B. */
template <class T> __device__ void blockDenseTrsmL (const T *L, T *B, const int nx_l, const int ny_l, const int nx_b, const int ld_l, const int ld_b, const int *pivot = nullptr)
{
  if (pivot != nullptr)
  { 
    blockApplyPivot(B, pivot, nx_b, ny_l, ld_b, false); 
  }
  for (int i = 0; i < nx_l && i + 1 < ny_l; i++)
  {
    blockDenseGemm <T> (-1.0, 1.0, &B[(i + 1) * ld_b], &L[(i + 1) * ld_a + i], &B[i * ld_b], ny_l - (i + 1), nx_b, 1, ld_b, ld_l, ld_b);
  }
}

/* U is ny_u x nx_u upper triangular and not unit diagonal, B is ny_b by nx_u, solves X x U = B, overwrites X in B. */
template <class T> __device__ void blockDenseTrsmR (const T *U, T *B, const int nx_u, const int ny_u, const int ny_b, const int ld_u, const int ld_b)
{
  for (int i = 0; i < nx_u && i < ny_u; i++)
  {
    blockDenseScalar <T> (-1.0 / U[i * ld_u + i], &B[i], 1, ny_b, ld_b);
    if (nx_u - i > 1)
    {
      blockDenseGemm <T> (-1.0, 1.0, &B[i + 1], &B[i], &U[i * ld_u + (i + 1)], ny_b, nx_u - (i + 1), 1, ld_b, ld_b, ld_u);
    }
  }
}


#endif