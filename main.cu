
#include <pspl.cuh>

template <class T> __global__ void kernel (inst_handler <T> ih, const int look_ahead_offset) 
{ 
  ih.run (look_ahead_offset);
}

template <class T> __host__ int test0()
{
  cudaSetDevice(0);
  cudaDeviceReset();

  const int n = 32, levels = 0, dim = 256;

  dev_hierarchical <T> *a = new dev_hierarchical <T> (n, n);
  a -> loadTestMatrix(levels, n, dim);
  //a->print();
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
  int look_ahead_offset = 32;

  myTimer.newEvent("GETRF", start, main_stream);
  cudaLaunchKernel((void *)kernel <T>, blocks, threads, (void **) new void *[2]{ ih, &look_ahead_offset }, 0, main_stream);
  myTimer.newEvent("GETRF", end, main_stream);

  fprintf(stderr, "Kernel Launch: %s\n\n", cudaGetErrorString(cudaGetLastError()));
  cudaError_t error = myTimer.dumpAllEvents_Sync(d -> getFops());
  delete d, ih;

  if (error == cudaSuccess)
  {
    dev_dense <T> *b = a -> convertToDense() -> restoreLU();
    a -> loadTestMatrix(levels, n, dim);
    //a->print();
    dev_dense <T> *c = a -> convertToDense();
    printf("Rel. L2 Error: %e\n\n", b -> L2Error(c));

    delete a, b, c;
  }

  cudaStreamDestroy(main_stream);
  return 0;
}

__global__ void svd_kernel (double * U, double *S, double * VT, const int nx, const int ny, const int rank)
{
  __shared__ double shm[256];
  int i = blockJacobiSVD <double> (U, VT, nx, ny, rank, rank, 1.0e-14, 100, &shm[0]);
  normalizeU(U, S, nx, ny, rank, rank);
  if (thread_rank() == 0) { printf("iters: %d\n", i); }
}

int test1()
{
  cudaSetDevice(0);
  cudaDeviceReset();

  const int nx = 16, ny = 16;

  dev_low_rank <double> *A = new dev_low_rank <double> (nx, ny);

  A -> getU() -> loadTestMatrix(20);

  timer myTimer = timer();

  myTimer.newEvent("SVD", start);
  svd_kernel <<<1, 1024 >>> (A -> getElementsU(), A -> getElementsS(), A->getElementsVT(), nx, ny, A -> getRank());
  myTimer.newEvent("SVD", end);

  myTimer.dumpAllEvents_Sync();

  A->print();

  dev_dense <double> *b = A->convertToDense(), *c = new dev_dense<double>(nx, ny);
  c->loadTestMatrix(20);
  printf("Rel. L2 Error: %e\n\n", c->L2Error(b));

  delete A, b, c;

  return 0;
}

__global__ void partial_pivot_kernel (double *matrix, const int nx, const int ny, const int ld, int *pivot)
{
  __shared__ double shm[4096];
  blockDenseGetrf_shm <double> (matrix, nx, ny, ld, pivot, &shm[0]);
}

__global__ void recover_pivot_kernel (double *matrix, const int nx, const int ny, const int ld, int *pivot)
{
  __shared__ double shm[4096];
  blockApplyPivot <double> (matrix, pivot, nx, ny, ld, true, &shm[0], 4096);
}

__host__ int test2()
{
  cudaSetDevice(0);
  cudaDeviceReset();
  const int nx = 4, ny = 4;

  dev_dense <double> *a = new dev_dense <double> (nx, ny, nx, true);
  a -> loadRandomMatrix(-10, 10, 999);
  a -> print();

  timer myTimer = timer();

  myTimer.newEvent("pivot", start);
  partial_pivot_kernel <<<1, 1024, 0, 0 >>> (a -> getElements(), nx, ny, nx, a -> getPivot());
  myTimer.newEvent("pivot", end);
  cudaDeviceSynchronize();

  a->print();
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
  //test0 <double> ();
  test1();
  //test2();

  return 0;
}