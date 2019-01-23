#ifndef DENSE_GETRF_CUH
#define DENSE_GETRF_CUH

#include <helper_functions.h>
#include <cuda_helper_functions.cuh>
#include <cub/cub.cuh>

__device__ void dense_getrf (double *matrix, const unsigned nx, const unsigned ld, const unsigned ny, const unsigned thread_id, const unsigned block_size)
{
  const unsigned n = min_(nx, ny);
  for (unsigned i = 0; i < n; i++)
  {
    const double diag = matrix[i * ld + i];
    const unsigned step_x = min_(nx - i, block_size / (ny - i)), step_y = min_(ny - i, block_size);
    const unsigned thread_x = min_(step_x, thread_id / step_y + 1), thread_y = thread_id - (thread_x - 1) * step_y + 1;

    #pragma unroll
    for (unsigned row = i + thread_y; row < ny; row += step_y) 
    {
      double left_col = matrix[row * ld + i] / diag;
      if (thread_x == 1) { matrix[row * ld + i] = left_col; }

      #pragma unroll
      for (unsigned col = i + thread_x; col < nx; col += step_x)
      {
        double top_row = matrix[i * ld + col];
        matrix[row * ld + col] -= left_col * top_row;
      }
    }

    __syncthreads();
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
    dense_getrf(matrix, nx, ld, ny, thread_id, block_size);
  }
  else /* matrix is small enough to load all in shared memory. */
  {
    extern __shared__ double shm_matrix[];

    #pragma unroll
    for (unsigned i = thread_id; i < nx * ny; i += block_size)
    { 
      const unsigned row = i / nx, col = i - row * nx, index = row * ld + col;
      shm_matrix[i] = matrix[index];
    }
    __syncthreads();
  
    dense_getrf(&shm_matrix[0], nx, nx, ny, thread_id, block_size);
  
    #pragma unroll
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

  matrix_copy_toDevice_sync (matrix, &dev_matrix, nx, ld, ny);
  create_timing_event_to_stream ("GETRF TOTAL", main_stream);

  dense_getrf_kernel2 <<<grid, block, shm_size, main_stream>>> (dev_matrix, nx, ld_aligned, ny);

  create_timing_event_to_stream ("GETRF TOTAL", main_stream);
  cudaStreamDestroy(main_stream);

  device_sync_dump_timed_events ();
  printf("Cuda Execution: getrf finished.\n\n");

  matrix_copy_toHost_sync (&dev_matrix, matrix, nx, ld, ny, true);
  cudaDeviceReset();
  return 0;
}
#endif