
#ifndef _DEV_DENSE_FUNCS_CUH
#define _DEV_DENSE_FUNCS_CUH

#include <pspl.cuh>

template <class T>
__device__ void blockDenseScalar (const T scale, T *matrix, const int nx, const int ny, const int ld)
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

template <class T>
__device__ void blockDenseGemm (const T alpha, const T beta, T *matrix, const T *a, const T *b, const int m, const int n, const int k, 
  const int ld_m, const int ld_a, const int ld_b)
{
  /* A has dimension m * k, B has dimension k * n, matrix has dimension m * n. matrix = alpha * A * B + beta * old_matrix. */
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

template <class T>
__device__ void blockDenseGetrf (T *matrix, const int nx, const int ny, const int ld, int *pivot = nullptr)
{
  if (pivot != nullptr) { resetPivot(pivot, ny); }

  const int n = (nx < ny) ? nx : ny;
  for (int i = 0; i < n; i++)
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

template <class T>
__device__ void blockDenseGessm(const T *A, T *B, const int m, const int n, const int ld_a, const int ld_b, const int *pivot = nullptr)
{
  for (int i = 0; i < m; i++)
  {
    if (i != m - 1)
    {
      blockDenseGemm <T>(-1.0, 1.0, &B[(i + 1) * ld_b], &A[(i + 1) * ld_a + i], &B[i * ld_b], m - (i + 1), n, 1, ld_b, ld_a, ld_b);
    }
  }
}


#endif