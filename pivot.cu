#include <pivot.cuh>

#include <helper_functions.h>
#include <cuda_helper_functions.cuh>

__global__ void partial_pivot_kernel (unsigned int *pivot, double *matrix, const unsigned int nx, const unsigned int ld, const unsigned int ny)
{
  thread_block g = this_thread_block();
  for (unsigned int i = g.thread_rank(); i < ny; i += g.size()) { pivot[i] = i; }

  const unsigned int n = min_(nx, ny);
  for (unsigned int i = 0; i < n; i++)
  {
    unsigned int target = blockAllFindRowPivot <double, 32> (i, matrix, nx, ld, ny);

    blockExchangeRow <double> (g, i, target, pivot, matrix, nx, ld, ny);
  }

  blockApplyPivot <double> (g, pivot, true, matrix, nx, ld, ny);
  blockApplyPivot <double> (g, pivot, false, matrix, nx, ld, ny);
  blockApplyPivot <double> (g, pivot, true, matrix, nx, ld, ny);
}

__host__ int main()
{
  cudaSetDevice(0);
  const unsigned nx = 16, ld = 16, ny = 16;
  double *matrix = randomMatrix(nx, ny, 0, 10);
  unsigned *pivot = (unsigned*) malloc(ny * sizeof(unsigned));
  printMatrix(matrix, nx, ld, ny);

  double *dev_matrix = 0;
  unsigned *dev_pivot = 0;
  matrix_copy_toDevice_sync <double> (matrix, &dev_matrix, nx, ld, ny);
  matrix_copy_toDevice_sync <unsigned> (pivot, &dev_pivot, ny, ny, 1);

  dim3 block(BLOCK_SIZE, BLOCK_SIZE), grid(1);
  create_timing_event_to_stream ("pivot", 0);
  partial_pivot_kernel <<<grid, block>>> (dev_pivot, dev_matrix, nx, ld, ny);
  create_timing_event_to_stream ("pivot", 0);
  device_sync_dump_timed_events ();

  matrix_copy_toHost_sync <double> (&dev_matrix, matrix, nx, ld, ny, true);
  matrix_copy_toHost_sync <unsigned> (&dev_pivot, pivot, ny, ny, 1, true);

  printMatrix(matrix, nx, ld, ny);
  printf("\n");
  for(unsigned x = 0; x < nx; x++)
  {
    printf("%d, ", pivot[x]);
  }
  printf("\n");


  cudaDeviceReset();
  return 0;
}