
#include <pspl.cuh>

template <class T> __global__ void kernel (inst_handler <T> ih, const int look_ahead_offset) 
{ 
  ih.run (look_ahead_offset);
}

template <class T> __host__ int test0()
{
  cudaSetDevice(0);
  cudaDeviceReset();

  const int n = 3, levels = 2, dim = 32;

  dev_hierarchical <T> *a = new dev_hierarchical <T> (n, n);
  a -> loadTestMatrix(levels, n, dim);
  printf("Testing: %d x %d.\n", a -> getNy(), a -> getNx());

  dev_dense <T> *c = new dev_dense<T> (a -> getNx(), a -> getNy());
  c -> loadTestMatrix();
  printf("Converted to Dense.\n");

  const h_ops_tree *tree = a -> generateOps_GETRF();
  //tree->print();

  h_ops_dag *d = new h_ops_dag (tree);
  //d->print();
  delete tree;

  inst_handler <T> * ih = new inst_handler <T> (d, a);
  delete d;

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
  cudaLaunchKernel((void *)kernel <T>, blocks, threads, (void **) new void *[2]{ ih, &look_ahead_offset }, 0, main_stream);
  myTimer.newEvent("GETRF", end, main_stream);

  fprintf(stderr, "Kernel Launch: %s\n\n", cudaGetErrorString(cudaGetLastError()));
  cudaError_t error = myTimer.dumpAllEvents_Sync(d -> getFops());

  if (error == cudaSuccess)
  {
    dev_dense <T> *b = a -> convertToDense(), *b_ = b -> restoreLU();
    delete b;

    printf("Rel. L2 Error: %e\n\n", b_ -> L2Error(c));

    delete b_;
  }
  delete ih;
  delete a;
  delete c;


  cudaStreamDestroy(main_stream);
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
  blockDenseGetrf_shm <double> (matrix, nx, ny, ld, pivot, &shm[0]);
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

__global__ void gemm_kernel(double *mat, double *a, double *b, double *c, double *d, const int m, const int n, const int k, const int l, const int o,
  const int ld_m, const int ld_a, const int ld_b, const int ld_c, const int ld_d)
{
  __shared__ double shm[6144];
  blockDenseGemm_4x_Cshm_RM_Set <double> (mat, a, b, c, d, m, n, k, l, o, ld_m, ld_a, ld_b, ld_c, ld_d, true, true, true, true, &shm[0], 6144);
}
  
__host__ int test3()
{
  cudaSetDevice(0);
  cudaDeviceReset();

  const int m = 1000, n = 1016, k = 998, l = 1020, o = 1024;
  dev_dense <double> *mat = new dev_dense<double>(n, m);
  dev_dense <double> *a = new dev_dense<double>(m, k);
  dev_dense <double> *b = new dev_dense<double>(k, l);
  dev_dense <double> *c = new dev_dense<double>(l, o);
  dev_dense <double> *d = new dev_dense<double>(o, n);
  
  a->loadRandomMatrix(-10, 10);
  b->loadRandomMatrix(-10, 10);
  c->loadRandomMatrix(-10, 10);
  d->loadRandomMatrix(-10, 10);

  
  timer myTimer = timer();
  
  myTimer.newEvent("gemm", start);
  gemm_kernel <<<1, 1024, 0, 0 >>> (mat->getElements(), a->getElements(), b->getElements(), c->getElements(), d->getElements(),
    m, n, k, l, o, mat->getLd(), a->getLd(), b->getLd(), c->getLd(), d->getLd());
  myTimer.newEvent("gemm", end);
  
  myTimer.printStatus();
  myTimer.dumpAllEvents_Sync();
  
  dev_dense <double> *e = a->transpose()->matrixMultiplication(b->transpose()) ->matrixMultiplication(c->transpose())->matrixMultiplication(d->transpose());
  printf("Rel. L2 Error: %e\n\n", e->L2Error(mat));
  
  delete mat;
  delete a; delete b; delete c; delete d; delete e;
  return 0;
}

template <class T> __host__ int test4()
{
  cudaSetDevice(0);
  cudaDeviceReset();

  const int n = 2, levels = 1, dim = 4, rank = 2;

  dev_hierarchical <T> *a = new dev_hierarchical <T> (n, n);
  a->loadTestMatrix2 (levels, n, dim, rank);

  const h_ops_tree *tree = a -> generateOps_GETRF();
  tree->print();

  delete tree;

  delete a;
  return 0;
}


int main(int argc, char **argv)
{
  //test0 <double> ();
  //test1();
  //test2();
  //test3();
  test4<double>();
  return 0;
}