#ifndef DENSE_GETRF_CUH
#define DENSE_GETRF_CUH

#include <pivot.cuh>

using namespace cooperative_groups;

template <class matrixEntriesT>
__device__ void blockDenseScalar (thread_group g, const matrixEntriesT scale, matrixEntriesT *matrix, const unsigned int nx, const unsigned int ld, const unsigned int ny)
{
  for (unsigned int i = g.thread_rank(); i < nx * ny; i += g.size())
  { 
    const unsigned row = i / nx, col = i - row * nx;
    matrix[row * ld + col] = (scale == 0) ? 0 : matrix[row * ld + col] * scale;
  }
  g.sync();
}

template <class matrixEntriesT>
__device__ void blockDenseGemm (thread_group g, const matrixEntriesT alpha, const matrixEntriesT beta, matrixEntriesT *a, matrixEntriesT *b, matrixEntriesT *matrix, 
  const unsigned int ld_a, const unsigned int ld_b, const unsigned int ld_m, const unsigned int m, const unsigned int n, const unsigned int k)
{
  /* A has dimension m * k, B has dimension k * n, matrix has dimension m * n. matrix = alpha * A * B + beta * old_matrix. */
  for (unsigned int i = g.thread_rank(); i < m * n; i += g.size())
  { 
    const unsigned row = i / n, col = i - row * n;
    const matrixEntriesT old = (beta == 0) ? 0 : beta * matrix[row * ld_m + col];
    matrixEntriesT accum = 0;
    if (alpha != 0)
    {
      for (unsigned int j = 0; j < k; j++)
      { accum += a[row * ld_a + j] * b[j * ld_b + col]; }
      accum *= alpha;
    }
    matrix[row * ld_m + col] = old + accum;
  }
  g.sync();
}

template <class matrixEntriesT>
__device__ void blockDenseGetrfNoPivot (matrixEntriesT *matrix, const unsigned int nx, const unsigned int ld, const unsigned int ny)
{
  const thread_block g = this_thread_block();
  const unsigned int n = (nx < ny) ? nx : ny;
  for (unsigned int i = 0; i < n; i++)
  {
    blockDenseScalar <matrixEntriesT> (g, 1.0 / matrix[i * ld + i], &matrix[(i + 1) * ld + i], 1, ld, ny - (i + 1));
    
    blockDenseGemm <matrixEntriesT> (g, -1.0, 1.0, &matrix[(i + 1) * ld + i], &matrix[i * ld + (i + 1)], &matrix[(i + 1) * ld + (i + 1)], 
      ld, ld, ld, ny - (i + 1), nx - (i + 1), 1);
  }
}

template <class matrixEntriesT, unsigned int tile_size>
__device__ void blockDenseGetrfWithPivot (unsigned int *pivot, matrixEntriesT *matrix, const unsigned int nx, const unsigned int ld, const unsigned int ny)
{
  const thread_block g = this_thread_block();
  for (unsigned int i = g.thread_rank(); i < ny; i += g.size()) { pivot[i] = i; }

  const unsigned int n = (nx < ny) ? nx : ny;
  for (unsigned int i = 0; i < n; i++)
  {
    unsigned int target = blockAllFindRowPivot <matrixEntriesT, tile_size> (i, matrix, nx, ld, ny);
    blockExchangeRow <matrixEntriesT> (g, i, target, pivot, matrix, nx, ld, ny);

    blockDenseScalar <matrixEntriesT> (g, 1.0 / matrix[i * ld + i], &matrix[(i + 1) * ld + i], 1, ld, ny - (i + 1));

    blockDenseGemm <matrixEntriesT> (g, -1.0, 1.0, &matrix[(i + 1) * ld + i], &matrix[i * ld + (i + 1)], &matrix[(i + 1) * ld + (i + 1)], 
      ld, ld, ld, ny - (i + 1), nx - (i + 1), 1);
  }
}

#endif