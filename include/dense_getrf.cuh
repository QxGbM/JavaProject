#ifndef DENSE_GETRF_CUH
#define DENSE_GETRF_CUH

#include <helper_functions.h>
#include <cuda_helper_functions.cuh>
#include <cooperative_groups.h>

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
}

template <class matrixEntriesT>
__device__ void blockDenseGemm (thread_group g, const matrixEntriesT alpha, const matrixEntriesT beta, matrixEntriesT *a, matrixEntriesT *b, matrixEntriesT *matrix, 
  const unsigned int ld_a, const unsigned int ld_b, const unsigned int ld_m, const unsigned int m, const unsigned int n, const unsigned int k)
{
  /* A has dimension m * k, B has dimension k * n, matrix has dimension m * n. matrix = alpha * A * B + beta * old_matrix. */
  for (unsigned int i = g.thread_rank(); i < m * n; i += g.size())
  { 
    const unsigned row = i / n, col = i - row * n;
    matrixEntriesT old = (beta == 0) ? 0 : beta * matrix[row * ld_m + col];
    matrixEntriesT accum = 0;
    if (alpha != 0)
    {
      for (unsigned int j = 0; j < k; j++)
      { accum += a[row * ld_a + j] * b[j * ld_b + col]; }
      accum *= alpha;
    }
    matrix[row * ld_m + col] = old + accum;
  }
}

template <class matrixEntriesT>
__device__ void blockDenseGetrfNoPivot (matrixEntriesT *matrix, const unsigned int nx, const unsigned int ld, const unsigned int ny)
{
  thread_block g = this_thread_block();
  const unsigned int n = min_(nx, ny);
  for (unsigned int i = 0; i < n; i++)
  {
    blockDenseScalar (g, 1.0 / matrix[i * ld + i], &matrix[(i + 1) * ld + i], 1, ld, ny - (i + 1));
    g.sync();

    blockDenseGemm (g, -1.0, 1.0, &matrix[(i + 1) * ld + i], &matrix[i * ld + (i + 1)], &matrix[(i + 1) * ld + (i + 1)], 
      ld, ld, ld, ny - (i + 1), nx - (i + 1), 1);
    g.sync();
  }
}

template <class matrixEntriesT, unsigned int tile_size>
__device__ void blockDenseGetrfWithPivot (unsigned int *pivot, matrixEntriesT *matrix, const unsigned int nx, const unsigned int ld, const unsigned int ny)
{
  thread_block g = this_thread_block();
  for (unsigned int i = g.thread_rank(); i < ny; i += g.size()) { pivot[i] = i; }

  const unsigned int n = min_(nx, ny);
  for (unsigned int i = 0; i < n; i++)
  {
    unsigned int target = blockAllFindRowPivot (i, matrix, nx, ld, ny);
    blockExchangeRow (g, i, target, pivot, matrix, nx, ld, ny);
    g.sync();

    blockDenseScalar (g, 1.0 / matrix[i * ld + i], &matrix[(i + 1) * ld + i], 1, ld, ny - (i + 1));
    g.sync();

    blockDenseGemm (g, -1.0, 1.0, &matrix[(i + 1) * ld + i], &matrix[i * ld + (i + 1)], &matrix[(i + 1) * ld + (i + 1)], 
      ld, ld, ld, ny - (i + 1), nx - (i + 1), 1);
    g.sync();
  }
}

__global__ void dense_getrf_kernel2 (double *matrix, const unsigned nx, const unsigned ld, const unsigned ny)
{
  /* 
  * Using 1 block, running parallel both horizontal and vertical 
  */
  const unsigned thread_id = threadIdx.x, block_size = blockDim.x;

  if (nx * ny > 6 * 1024) /* matrix is too big to load all in shared memory. */
  {
    blockDenseGetrfNoPivot <double> (matrix, nx, ld, ny);
  }
  else /* matrix is small enough to load all in shared memory. */
  {
    extern __shared__ double shm_matrix[];

    for (unsigned i = thread_id; i < nx * ny; i += block_size)
    { 
      const unsigned row = i / nx, col = i - row * nx, index = row * ld + col;
      shm_matrix[i] = matrix[index];
    }
    __syncthreads();
  
    blockDenseGetrfNoPivot <double> (&shm_matrix[0], nx, nx, ny);

    for (unsigned i = thread_id; i < nx * ny; i += block_size)
    { 
      const unsigned row = i / nx, col = i - row * nx, index = row * ld + col;
      matrix[index] = shm_matrix[i];
    }
  }

}

__host__ int dense_getrf_sync2 (Matrix *m) 
{
  double *matrix = m -> head;
  const unsigned nx = m -> nx, ld = m -> ld, ny = m -> ny;
  if (ld < nx) { printf("GETRF ABORT: Matrix's horizontal offset is less than the number of entries.\n");  return -1; }

  double *dev_matrix = 0;

  cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

  dim3 block(1024), grid(1);
  cudaStream_t main_stream;
  cudaStreamCreate(&main_stream);

  const unsigned ld_aligned = ((ld + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
  const unsigned shm_size = (nx * ny > 6 * 1024) ? 0 : nx * ny * sizeof(double);
  if (shm_size == 0) { printf("WARNING: Matrix size exceeded 48KB of shared memory size. \n-------- Using Global mem instead.\n\n"); }

  matrix_copy_toDevice_sync <double> (matrix, &dev_matrix, nx, ld, ny);
  create_timing_event_to_stream ("GETRF TOTAL", main_stream);

  dense_getrf_kernel2 <<<grid, block, shm_size, main_stream>>> (dev_matrix, nx, ld_aligned, ny);

  create_timing_event_to_stream ("GETRF TOTAL", main_stream);
  cudaStreamDestroy(main_stream);

  device_sync_dump_timed_events ();
  printf("Cuda Execution: getrf finished.\n\n");

  matrix_copy_toHost_sync <double> (&dev_matrix, matrix, nx, ld, ny, true);
  cudaDeviceReset();
  return 0;
}
#endif