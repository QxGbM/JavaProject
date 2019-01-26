#include <pivot.cuh>
#include <dense_getrf.cuh>

#include <helper_functions.h>
#include <cuda_helper_functions.cuh>

__global__ void partial_pivot_kernel (unsigned int *pivot, double *matrix, const unsigned int nx, const unsigned int ld, const unsigned int ny)
{
  blockDenseGetrfWithPivot <double, 32> (pivot, matrix, nx, ld, ny);
}

__global__ void recover_pivot_kernel (unsigned int *pivot, double *matrix, const unsigned int nx, const unsigned int ld, const unsigned int ny)
{
  blockApplyPivot <double> (this_thread_block(), pivot, true, matrix, nx, ld, ny);
}

__host__ int main()
{
  cudaSetDevice(0);
  const unsigned nx = 20, ld = 20, ny = 20;
  double *matrix = randomMatrix(nx, ny, 0, 10);
  unsigned *pivot = (unsigned*) malloc(ny * sizeof(unsigned));
  printMatrix(matrix, nx, ld, ny);

  double *dev_matrix = 0;
  unsigned *dev_pivot = 0;
  matrix_copy_toDevice_sync <double> (matrix, &dev_matrix, nx, ld, ny);
  matrix_copy_toDevice_sync <unsigned> (pivot, &dev_pivot, ny, ny, 1);

  dim3 block(256), grid(1);
  create_timing_event_to_stream ("pivot", 0);
  partial_pivot_kernel <<<grid, block>>> (dev_pivot, dev_matrix, nx, ld, ny);
  create_timing_event_to_stream ("pivot", 0);
  device_sync_dump_timed_events ();

  matrix_copy_toHost_sync <double> (&dev_matrix, matrix, nx, ld, ny, false);

  double *result = multiplyLU(matrix, nx, nx, ny);
  printMatrix(result, nx, ld, ny);
  double *dev_result = 0;

  matrix_copy_toDevice_sync <double> (result, &dev_result, nx, ld, ny);

  create_timing_event_to_stream ("pivot recovery", 0);
  recover_pivot_kernel <<<grid, block>>> (dev_pivot, dev_result, nx, ld, ny);
  create_timing_event_to_stream ("pivot recovery", 0);
  device_sync_dump_timed_events ();

  matrix_copy_toHost_sync <double> (&dev_result, result, nx, ld, ny, true);
  matrix_copy_toHost_sync <unsigned> (&dev_pivot, pivot, ny, ny, 1, true);

  printMatrix(result, nx, ld, ny);
  printf("\n");
  for(unsigned x = 0; x < nx; x++)
  {
    printf("%d, ", pivot[x]);
  }
  printf("\n");

  cudaDeviceReset();
  return 0;
}