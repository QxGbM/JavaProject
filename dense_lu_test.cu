

#include <dev_dense.cuh>
#include <cuda_timer.cuh>
#include <dense_getrf.cuh>

#define CUDA_DEVICE 0

__global__ void dense_getrf_kernel (double *matrix, const int nx, const int ld, const int ny)
{
  /* 
  * Using 1 block, running parallel both horizontal and vertical 
  */
  const int  thread_id = threadIdx.x, block_size = blockDim.x;

  if (nx * ny > 6 * 1024) /* matrix is too big to load all in shared memory. */
  {
    blockDenseGetrfNoPivot <double> (matrix, nx, ld, ny);
  }
  else /* matrix is small enough to load all in shared memory. */
  {
    extern __shared__ double shm_matrix[];

    for (int i = thread_id; i < nx * ny; i += block_size)
    { 
      const int row = i / nx, col = i - row * nx, index = row * ld + col;
      shm_matrix[i] = matrix[index];
    }
    __syncthreads();
  
    blockDenseGetrfNoPivot <double> (&shm_matrix[0], nx, nx, ny);

    for (int i = thread_id; i < nx * ny; i += block_size)
    { 
      const int row = i / nx, col = i - row * nx, index = row * ld + col;
      matrix[index] = shm_matrix[i];
    }
  }

}

__host__ int dense_getrf_sync (double *matrix, const int nx, const int ld, const int ny) 
{
  if (ld < nx) { printf("GETRF ABORT: Matrix's horizontal offset is less than the number of entries.\n");  return -1; }
  cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

  dim3 block(256), grid(1);
  cudaStream_t main_stream;
  cudaStreamCreate(&main_stream);

  const int shm_size = (nx * ny > 6 * 1024) ? 0 : nx * ny * sizeof(double);
  if (shm_size == 0) { printf("WARNING: Matrix size exceeded 48KB of shared memory size. \n-------- Using Global mem instead.\n\n"); }

  struct timer myTimer = timer();
  myTimer.newEvent("GETRF", main_stream);

  //create_timing_event_to_stream ("GETRF TOTAL", main_stream);

  dense_getrf_kernel <<<grid, block, shm_size, main_stream>>> (matrix, nx, ld, ny);

  //create_timing_event_to_stream ("GETRF TOTAL", main_stream);
  myTimer.newEvent("GETRF", main_stream);
  cudaStreamDestroy(main_stream);

  myTimer.printStatus();
  myTimer.dumpAllEvents_Sync();

  //device_sync_dump_timed_events ();
  printf("Cuda Execution: getrf finished.\n\n");

  return 0;
}

void test_dense_getrf_nxn (const int nx, const int ny)
{
  printf("-------- Testing %d x %d Dense GETRF: --------\n\n", ny, nx);
  cudaSetDevice(CUDA_DEVICE);
  printf("Running on cuda device: %d\n\n", CUDA_DEVICE);

  struct dev_dense <double> *a = new dev_dense <double> (nx, ny);
  a -> loadTestMatrix();
  a -> copyToDevice_Sync();

  dense_getrf_sync(a -> dev_ptr, a -> nx, a -> ld, a -> ny);

  a -> copyToHost_Sync();
  struct dev_dense <double> *b = a -> restoreLU();
  a -> loadTestMatrix();

  printf("Rel. L2 Error: %e\n", b -> L2Error(a));
  printf("-------- n x n Dense GETRF test finished --------\n\n");

  a -> ~dev_dense();
  b -> ~dev_dense();
  free(a);
  free(b);

}
/*
__global__ void dense_trsm_kernel(double *matrix, const  nx, const  ld, const  ny)
{
  
}

__host__ void dense_trsm_sync (Matrix *b, Matrix *a, const double alpha, const bool side, const bool unit_triangular, const bool uplo_lower)
{
  
}

extern "C" void test_inverse (const  nx, const  ny)
{
  cudaSetDevice(CUDA_DEVICE);
  printf("Running on cuda device: %d\n", CUDA_DEVICE);

  Matrix a = testMatrix_M(nx, ny);
  dense_getrf_sync(&a);

  Matrix b = identityMatrix_M(ny, ny);
  //dense_trsm_sync (&b, &a, 1, false, false, true, true);

  Matrix c = identityMatrix_M(nx, nx);
  //dense_trsm_sync (&c, &a, 1, true, false, true, true);

  Matrix result0 = matrixMultiplication(testMatrix_M(nx, ny), b);
  Matrix result1 = matrixMultiplication(c, testMatrix_M(nx, ny));

  printf("left inverse: Rel. L2 Error: %e\n", L2Error(result0, identityMatrix_M(ny, ny)));
  printf("right inverse: Rel. L2 Error: %e\n", L2Error(result1, identityMatrix_M(nx, nx)));
}*/


int main(int argc, char **argv)
{
  const int nx = 16;
  const int ny = 16;

  test_dense_getrf_nxn (nx, ny);

  //test_inverse (nx, ny);


  return 0;
}