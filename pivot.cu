#include <pivot.cuh>

//template __global__ void partial_pivot_kernel2 <double, 32> (matrixEntriesT *matrix, const unsigned nx, const unsigned ld, const unsigned ny, unsigned *p);

__host__ int main()
{
  cudaSetDevice(0);
  const unsigned nx = 16, ld = 16, ny = 16;
  //double test[16] = {2, 1, 1, 0, 4, 3, 3, 1, 8, 7, 9, 5, 6, 7, 9, 8};
  double *matrix = randomMatrix(nx, ny, 0, 10);
  //double *matrix = &test[0];
  printMatrix(matrix, nx, ld, ny);
  double *dev_matrix = 0;
  matrix_copy_toDevice_sync (matrix, &dev_matrix, nx, ld, ny);

  dim3 block(BLOCK_SIZE, BLOCK_SIZE), grid(1);
  create_timing_event_to_stream ("pivot", 0);
  //partial_pivot_kernel <<<grid, block>>> (dev_matrix, nx, ld, ny, nullptr);
  partial_pivot_kernel2 <<<grid, block>>> (dev_matrix, nx, ld, ny, nullptr);
  create_timing_event_to_stream ("pivot", 0);
  device_sync_dump_timed_events ();
  cudaDeviceReset();

  return 0;
}