
#include <pspl.cuh>

template <class T> __global__ void kernel (inst_handler <T> ih, const int look_ahead_offset) 
{ 
  ih.run (look_ahead_offset);
}

template <class T> __host__ int test0()
{
  cudaSetDevice(0);
  cudaDeviceReset();

  const int n = 16, levels = 0, dim = 64;

  dev_hierarchical <T> *a = new dev_hierarchical <T> (n, n);
  a -> loadTestMatrix(levels, n, dim);
  printf("Testing: %d x %d.\n", a -> getNy(), a -> getNx());

  h_ops_dag *d = new h_ops_dag (a -> generateOps_GETRF());

  inst_handler <T> * ih = new inst_handler <T> (d, a);

  timer myTimer = timer();
  cudaStream_t main_stream;
  cudaStreamCreate(&main_stream);

  if (sizeof(T) == 8) 
  {
    printf("shared mem double precision.\n"); 
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  }
  
  const int blocks = 64, threads = 1024;
  int look_ahead_offset = 64;

  myTimer.newEvent("GETRF", start, main_stream);
  cudaLaunchKernel((void *)kernel <T>, blocks, threads, new void *[2]{ ih, &look_ahead_offset }, 0, main_stream);
  myTimer.newEvent("GETRF", end, main_stream);

  fprintf(stderr, "Kernel Launch: %s\n\n", cudaGetErrorString(cudaGetLastError()));
  cudaError_t error = myTimer.dumpAllEvents_Sync(d -> getFops());
  delete d, ih;

  if (error == cudaSuccess)
  {
    dev_dense <T> *b = a -> convertToDense() -> restoreLU();
    a -> loadTestMatrix(levels, n, dim);
    dev_dense <T> *c = a -> convertToDense();
    printf("Rel. L2 Error: %e\n\n", b -> L2Error(c));
    delete a, b, c;
  }

  cudaStreamDestroy(main_stream);
  return 0;
}

__global__ void svd_kernel(double * A, double * VT, const int nx, const int ny, const int ld_a, const int ld_v)
{
  int i = blockJacobiSVD <double>(A, VT, nx, ny, ld_a, ld_v, 1.0e-14, 100);
  if (thread_rank() == 0) { printf("iters: %d\n", i); }
}

int test1()
{
  cudaSetDevice(0);
  cudaDeviceReset();

  const int nx = 16, ny = 16;

  dev_dense <double> *d_VT, *d_A;

  d_A = new dev_dense <double>(nx, ny);
  d_A->loadTestMatrix(20);

  d_VT = new dev_dense <double>(nx, nx);
  //d_VT -> loadIdentityMatrix();

  double *A = d_A->getElements();
  double *VT = d_VT->getElements();

  timer myTimer = timer();

  myTimer.newEvent("SVD", start);
  svd_kernel <<<1, 1024 >>> (A, VT, nx, ny, nx, nx);
  myTimer.newEvent("SVD", end);

  myTimer.dumpAllEvents_Sync();

  for (int i = 0; i < nx; i++)
  {
    double s = 0.0;
    for (int j = 0; j < ny; j++)
    {
      s += A[j * nx + i] * A[j * nx + i];
    }

    s = sqrt(s);
    printf("%d: %e\n", i, s);
  }

  dev_dense <double> *c = d_A->matrixMultiplication(d_VT->transpose());
  d_A->loadTestMatrix(20);
  printf("Rel. L2 Error: %e\n\n", c->L2Error(d_A));

  return 0;
}

__global__ void partial_pivot_kernel (double *matrix, const int nx, const int ny, const int ld, int *pivot)
{
  blockDenseGetrf_shm <double, 1024> (matrix, nx, ny, ld, pivot);
}

__global__ void recover_pivot_kernel (double *matrix, const int nx, const int ny, const int ld, int *pivot)
{
  blockApplyPivot <double, 512> (matrix, pivot, nx, ny, ld, true);
}

__host__ int test2()
{
  cudaSetDevice(0);
  cudaDeviceReset();
  const int nx = 256, ny = 128;

  dev_dense <double> *a = new dev_dense <double> (nx, ny, nx, true);
  a -> loadRandomMatrix(-10, 10, 999);

  timer myTimer = timer();

  myTimer.newEvent("pivot", start);
  partial_pivot_kernel <<<1, 1024, 0, 0 >>> (a -> getElements(), nx, ny, nx, a -> getPivot());
  myTimer.newEvent("pivot", end);
  cudaDeviceSynchronize();


  dev_dense <double> *b = a -> restoreLU();

  myTimer.newEvent("pivot recovery", start);
  recover_pivot_kernel <<<1, 1024, 0, 0 >>> (b -> getElements(), nx, ny, nx, a->getPivot());
  myTimer.newEvent("pivot recovery", end);

  myTimer.printStatus();
  myTimer.dumpAllEvents_Sync();

  a->loadRandomMatrix(-10, 10, 999);
  printf("Rel. L2 Error: %e\n\n", b -> L2Error(a));

  delete a;
  delete b;

  return 0;
}


int main(int argc, char **argv)
{
  test0 <double> ();
  test1();
  test2();

  return 0;
}