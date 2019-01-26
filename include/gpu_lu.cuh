#ifndef GPU_LU_CUH
#define GPU_LU_CUH

#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <cuda.h>

#include <helper_functions.h>
#include <dense_lu.cuh>
#include <dense_getrf.cuh>

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

extern "C" void test_dense_getrf_1x1 ()
{
  const unsigned nx = BLOCK_SIZE;
  const unsigned ny = BLOCK_SIZE;
  printf("-------- Testing single block Dense GETRF: --------\n\n");
  cudaSetDevice(CUDA_DEVICE);
  printf("Running on cuda device: %d\n\n", CUDA_DEVICE);

  Matrix a = testMatrix_M(nx, ny);
  dense_getrf_sync(&a);

  Matrix result = multiplyLU(a);

  printf("Rel. L2 Error: %e\n", L2Error(result, testMatrix_M(nx, ny)));
  printf("-------- 1 x 1 Dense GETRF test finished --------\n\n");

}

extern "C" void test_dense_getrf_nx1 (const unsigned ny)
{
  const unsigned nx = BLOCK_SIZE;
  printf("-------- Testing %d x %d Dense GETRF: --------\n\n", ny, nx);
  cudaSetDevice(CUDA_DEVICE);
  printf("Running on cuda device: %d\n\n", CUDA_DEVICE);

  Matrix a = testMatrix_M(nx, ny);
  dense_getrf_sync(&a);
  
  Matrix result = multiplyLU(a);

  printf("Rel. L2 Error: %e\n", L2Error(result, testMatrix_M(nx, ny)));
  printf("-------- n x 1 Dense GETRF test finished --------\n\n");

}

extern "C" void test_dense_getrf_1xn (const unsigned nx)
{
  const unsigned ny = BLOCK_SIZE;
  printf("-------- Testing %d x %d Dense GETRF: --------\n\n", ny, nx);
  cudaSetDevice(CUDA_DEVICE);
  printf("Running on cuda device: %d\n\n", CUDA_DEVICE);

  Matrix a = testMatrix_M(nx, ny);
  dense_getrf_sync(&a);
  
  Matrix result = multiplyLU(a);

  printf("Rel. L2 Error: %e\n", L2Error(result, testMatrix_M(nx, ny)));
  printf("-------- 1 x n Dense GETRF test finished --------\n\n");
}

extern "C" void test_dense_getrf_nxn (const unsigned nx, const unsigned ny)
{
  printf("-------- Testing %d x %d Dense GETRF: --------\n\n", ny, nx);
  cudaSetDevice(CUDA_DEVICE);
  printf("Running on cuda device: %d\n\n", CUDA_DEVICE);

  Matrix a = testMatrix_M(nx, ny);
  dense_getrf_sync2(&a);

  Matrix result = multiplyLU(a);

  printf("Rel. L2 Error: %e\n", L2Error(result, testMatrix_M(nx, ny)));
  printf("-------- n x n Dense GETRF test finished --------\n\n");

}

extern "C" void test_inverse (const unsigned nx, const unsigned ny)
{
  cudaSetDevice(CUDA_DEVICE);
  printf("Running on cuda device: %d\n", CUDA_DEVICE);

  Matrix a = testMatrix_M(nx, ny);
  dense_getrf_sync(&a);

  Matrix b = identityMatrix_M(ny, ny);
  dense_trsm_sync (&b, &a, 1, false, false, true, true);

  Matrix c = identityMatrix_M(nx, nx);
  dense_trsm_sync (&c, &a, 1, true, false, true, true);

  Matrix result0 = matrixMultiplication(testMatrix_M(nx, ny), b);
  Matrix result1 = matrixMultiplication(c, testMatrix_M(nx, ny));

  printf("left inverse: Rel. L2 Error: %e\n", L2Error(result0, identityMatrix_M(ny, ny)));
  printf("right inverse: Rel. L2 Error: %e\n", L2Error(result1, identityMatrix_M(nx, nx)));
}

extern "C" void test_dense_gemm_1block (const unsigned nx, const unsigned ny)
{
  cudaSetDevice(CUDA_DEVICE);
  printf("Running on cuda device: %d\n", CUDA_DEVICE);

  Matrix a = {testMatrix(nx, ny), nx, nx, ny};
  dense_getrf_sync(&a);

  Matrix b = {identityMatrix(ny, ny), ny, ny, ny};
  dense_trsm_sync (&b, &a, 1, false, false, true, true);

  Matrix c = {identityMatrix(nx, nx), nx, nx, nx};
  dense_trsm_sync (&c, &a, 1, true, false, true, true);

  Matrix d = {testMatrix(nx, ny), nx, nx, ny};
  double *t1 = (double*)malloc (ny * ny * sizeof(double));
  Matrix e = {t1, ny, ny, ny};
  dense_gemm_sync (&e, &d, &b, 1, 0);

  double *t2 = (double*)malloc (nx * nx * sizeof(double));
  Matrix f = {t2, nx, nx, nx};
  dense_gemm_sync (&f, &c, &d, 1, 0);

  double *result0 = e.head;
  double *result1 = f.head;

  printf("left inverse: Rel. L2 Error: %e\n", L2Error(result0, identityMatrix(ny, ny), ny, ny, ny));
  printf("right inverse: Rel. L2 Error: %e\n", L2Error(result1, identityMatrix(nx, nx), nx, nx, nx));

}


extern "C" void test_dense_getrf_2x2 (const unsigned nx, const unsigned ny)
{

  cudaSetDevice(CUDA_DEVICE);
  printf("Running on cuda device: %d\n", CUDA_DEVICE);

  double *matrix = testMatrix(nx, ny);

  Matrix a00 = {&matrix[0], nx / 2, nx, ny / 2};
  dense_getrf_sync(&a00);

  Matrix a01 = {&matrix[nx / 2], nx - nx / 2, nx, ny / 2};
  dense_trsm_sync (&a01, &a00, 1, false, true, true, false);

  Matrix a10 = {&matrix[nx * ny / 2], nx / 2, nx, ny - ny / 2};
  dense_trsm_sync (&a10, &a00, 1, true, false, false, true);

  Matrix a11 = {&matrix[nx * ny / 2 + nx / 2], nx - nx / 2, nx, ny - ny / 2};
  dense_gemm_sync (&a11, &a10, &a01, -1, 1);
  dense_getrf_sync(&a11);

  double *result = multiplyLU(matrix, nx, nx, ny);

  printf("Rel. L2 Error: %e\n", L2Error(result, testMatrix(nx, ny), nx, nx, ny));

}

#endif