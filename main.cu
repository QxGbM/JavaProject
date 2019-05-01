
#include <pspl.cuh>
#define ref

template <class T> __host__ int test0()
{
  cudaSetDevice(0);
  cudaDeviceReset();

  const int n = 2, levels = 2, dim = 64, rank = 32;

  dev_hierarchical <T> *a = new dev_hierarchical <T> (n, n);
  //a -> loadTestMatrix(levels, n, dim);
  a -> loadTestMatrix2(levels, n, dim, rank);

  const int blocks = 56, threads = 1024;

#ifdef ref
  dev_dense <T> *c = a -> convertToDense();
  printf("Reference Matrix converted to dense.\n");
#endif // ref

  cudaError_t error = hierarchical_GETRF <T, 12288> (a, blocks, threads);

#ifdef ref
  if (error == cudaSuccess)
  {
    dev_dense <T> *b = a -> convertToDense(), *b_ = b -> restoreLU();
    delete b;

    printf("Rel. L2 Error: %e\n\n", b_ -> L2Error(c));
    delete b_;
  }
  delete c;
#endif // ref

  delete a;

  return 0;
}

__global__ void svd_kernel (double * U, double * VT, const int nx, const int ny, const int ld_u, const int ld_v)
{
  __shared__ double shm[256];
  int i = blockJacobiSVD <double> (U, VT, nx, ny, ld_u, ld_v, 1.0e-14, 100, &shm[0]);
  if (thread_rank() == 0) { printf("iters: %d\n", i); }
}

int test1()
{
  cudaSetDevice(0);
  cudaDeviceReset();

  const int nx = 16, ny = 16;

  dev_low_rank <double> *A = new dev_low_rank <double> (nx, ny);

  A -> getUxS() -> loadTestMatrix(20);

  timer myTimer = timer();

  myTimer.newEvent("SVD", start);
  svd_kernel <<<1, 1024 >>> (A -> getElements(), A -> getElements(A -> getOffset_VT()), nx, ny, A -> getLd_UxS(), A -> getLd_VT());
  myTimer.newEvent("SVD", end);

  myTimer.dumpAllEvents_Sync();
  A->adjustRank(6);
  A->print();

  dev_dense <double> *b = A->convertToDense(), *c = new dev_dense<double>(nx, ny);
  c->loadTestMatrix(20);
  printf("Rel. L2 Error: %e\n\n", c->L2Error(b));

  delete A; delete b; delete c;

  return 0;
}

__global__ void partial_pivot_kernel (double *matrix, const int nx, const int ny, const int ld, int *pivot)
{
  __shared__ double shm[6144];
  blockDenseGetrf_shm <double> (matrix, pivot, nx, ny, ld, &shm[0]);
}

__global__ void recover_pivot_kernel (double *matrix, const int nx, const int ny, const int ld, int *pivot)
{
  __shared__ double shm[6144];
  blockApplyPivot <double> (matrix, pivot, nx, ny, ld, true, &shm[0], 6144);
}

__host__ int test2()
{
  cudaSetDevice(0);
  cudaDeviceReset();
  const int nx = 512, ny = 512;

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
  //test1();
  //test2();

  return 0;
}