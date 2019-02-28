
#ifndef _DENSE_GETRF_CUH
#define _DENSE_GETRF_CUH

#include <pivot.cuh>

using namespace cooperative_groups;

template <class matrixEntriesT>
__device__ void blockDenseScalar (const thread_group g, const matrixEntriesT scale, matrixEntriesT *matrix, 
  const int nx, const int ld, const int ny)
{
  for (int i = g.thread_rank(); i < nx * ny; i += g.size())
  { 
    const int row = i / nx, col = i - row * nx;
    matrix[row * ld + col] = (scale == 0) ? 0 : matrix[row * ld + col] * scale;
  }
  g.sync();
}

template <class matrixEntriesT>
__device__ void blockDenseGemm (const thread_group g, const matrixEntriesT alpha, const matrixEntriesT beta, const matrixEntriesT *a, 
  const matrixEntriesT *b, matrixEntriesT *matrix, const int ld_a, const int ld_b, const int ld_m, 
  const int m, const int n, const int k)
{
  /* A has dimension m * k, B has dimension k * n, matrix has dimension m * n. matrix = alpha * A * B + beta * old_matrix. */
  for (int i = g.thread_rank(); i < m * n; i += g.size())
  { 
    const int row = i / n, col = i - row * n;
    const matrixEntriesT old = (beta == 0) ? 0 : beta * matrix[row * ld_m + col];
    matrixEntriesT accum = 0;
    if (alpha != 0)
    {
      for (int j = 0; j < k; j++)
      { accum += a[row * ld_a + j] * b[j * ld_b + col]; }
      accum *= alpha;
    }
    matrix[row * ld_m + col] = old + accum;
  }
  g.sync();
}

template <class matrixEntriesT>
__device__ void blockDenseGessm (thread_group g, const matrixEntriesT *A, const int *pivot, matrixEntriesT *B, 
  const int m, const int n, const int ld_a, const int ld_b)
{
  for (int i = 0; i < m; i++)
  {
    //if (!diag_unit) { blockDenseScalar <matrixEntriesT> (g, 1.0 / A[i * ld_a + i], &B[i * ld_b], n, ld_b, 1); }
    if (i != m - 1)
    { blockDenseGemm <matrixEntriesT> (g, -1.0, 1.0, &A[(i + 1) * ld_a + i], &B[i * ld_b], &B[(i + 1) * ld_b], 
        ld_a, ld_b, ld_b, m - (i + 1), n, 1); }
  }
}


template <class matrixEntriesT>
__device__ void blockDenseGetrfNoPivot (matrixEntriesT *matrix, const int nx, const int ld, const int ny)
{
  const thread_block g = this_thread_block();
  const int n = (nx < ny) ? nx : ny;
  for (int i = 0; i < n; i++)
  {
    blockDenseScalar <matrixEntriesT> (g, 1.0 / matrix[i * ld + i], &matrix[(i + 1) * ld + i], 1, ld, ny - (i + 1));
    
    blockDenseGemm <matrixEntriesT> (g, -1.0, 1.0, &matrix[(i + 1) * ld + i], &matrix[i * ld + (i + 1)], &matrix[(i + 1) * ld + (i + 1)], 
      ld, ld, ld, ny - (i + 1), nx - (i + 1), 1);
  }
}

template <class matrixEntriesT>
__device__ void blockDenseGetrfWithPivot (matrixEntriesT *matrix, int *pivot, const int nx, const int ld, const int ny)
{
  const thread_block g = this_thread_block();
  resetPivot(pivot, ny);

  const int n = (nx < ny) ? nx : ny;
  for (int i = 0; i < n; i++)
  {
    const int target = i + blockAllFindRowPivot <matrixEntriesT> (&matrix[i * ld + i], ny - i, ld);

    if (target != i)
    {
      blockSwapNSeqElements <matrixEntriesT> (&matrix[target * ld], &matrix[i * ld], nx);
      if (g.thread_rank() == 0) { int t = pivot[target]; pivot[target] = pivot[i]; pivot[i] = t; }
      g.sync();
    }

    blockDenseScalar <matrixEntriesT> (g, 1.0 / matrix[i * ld + i], &matrix[(i + 1) * ld + i], 1, ld, ny - (i + 1));

    blockDenseGemm <matrixEntriesT> (g, -1.0, 1.0, &matrix[(i + 1) * ld + i], &matrix[i * ld + (i + 1)], &matrix[(i + 1) * ld + (i + 1)], 
      ld, ld, ld, ny - (i + 1), nx - (i + 1), 1);
  }
}

#endif