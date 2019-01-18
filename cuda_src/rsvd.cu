
#include <cublas_v2.h>
#include <kblas.h>
#include <batch_rand.h>
#include <batch_svd.h>

#include <helper_functions.h>
#include <cuda_helper_functions.cuh>

int main(int argc, char** argv)
{
  double *matrix = testMatrix(16, 16), *s = zeroMatrix(16, 16);
  kblasHandle_t handle;
  kblasRandState_t rand_state;
  
  kblasCreate(&handle);
  kblasInitRandState(handle, &rand_state, 16384 * 2, 0);

  double *dev_matrix = 0, *dev_s = 0;
  matrix_copy_toDevice_sync (matrix, &dev_matrix, 16, 16, 16);
  matrix_copy_toDevice_sync (s, &dev_s, 16, 16, 16);

  cudaDeviceReset();

  kblasAllocateWorkspace(handle);
  kblasDrsvd_batch_strided(handle, 16, 16, 17, dev_matrix, 16, 16, dev_s, 16, rand_state, 1);

  matrix_copy_toHost_sync (&dev_matrix, matrix, 16, 16, 16, true);
  matrix_copy_toHost_sync (&dev_s, s, 16, 16, 16, true);

  printMatrix(matrix, 16, 16, 16);
  printMatrix(s, 16, 16, 16);

  kblasDestroy(handle);

  return 0;
}
