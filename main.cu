
#include <pspl.cuh>
#define ref


__global__ void partial_pivot_kernel(double *matrix, const int nx, const int ny, const int ld, int *pivot)
{
  __shared__ double shm[6144];
  blockDenseGetrf_shm <double>(matrix, pivot, nx, ny, ld, &shm[0]);
}

__global__ void recover_pivot_kernel(double *matrix, const int nx, const int ny, const int ld, int *pivot)
{
  __shared__ double shm[6144];
  blockApplyPivot <double>(matrix, pivot, nx, ny, ld, true, &shm[0], 6144);
}

template <class T> __host__ int test0()
{
  cudaSetDevice(0);
  cudaDeviceReset();

  const int n = 2, levels = 1, dim = 4, admis = 3;

  dev_hierarchical <T> *a = new dev_hierarchical <T> (n, n);
  a -> loadTestMatrix(levels, n, dim, admis);

  const int blocks = 160, threads = 1024;

#ifdef ref
  dev_dense <T> *c = a -> convertToDense();
  printf("Reference Matrix converted to dense.\n");
#endif // ref

  cudaError_t error = hierarchical_GETRF <T, 12288> (a, blocks, threads);

#ifdef ref
  if (error == cudaSuccess)
  {
    dev_dense <T> *b = a -> convertToDense();
    partial_pivot_kernel <<<1, 1024, 0, 0 >>> (c -> getElements(), a -> getNx_abs(), a -> getNy_abs(), a -> getNx_abs(), nullptr);
    cudaDeviceSynchronize();

    printf("Rel. L2 Error: %e\n\n", b -> L2Error(c));
    delete b;
  }
  delete c;
#endif // ref

  delete a;

  return 0;
}

__global__ void svd_kernel (double * U, double * VT, const int nx, const int ny, const int ld_u, const int ld_v)
{
  __shared__ double shm[6144];
  int i = blockRandomizedSVD <double> (U, VT, nx, ny, ld_u, ld_v, 8, 1.0e-14, 100, shm, 6144);
  if (thread_rank() == 0) { printf("iters: %d\n", i); }
}

__global__ void qr_kernel (double* Q, double* R, const int nx, const int ny, const int ld_q, const int ld_r)
{
  __shared__ double shm[6144];
  matrixCopy_fromRM (R, Q, nx, ny, ld_r, ld_q, false);
  blockGivensRotation (R, nx, ny, ld_r);
  blockDenseTrsmR_shm (Q, R, nx, ny, nx, ld_q, ld_r, false, shm, 6144);
  blockGramSchmidt (Q, nx, ny, ld_q, shm);

}

int test1()
{
  cudaSetDevice(0);
  cudaDeviceReset();

  const int nx = 32, ny = 512;

  srand(200);
  double * rnd_seed = new double[_RND_SEED_LENGTH];
#pragma omp parallel for
  for (int i = 0; i < _RND_SEED_LENGTH; i++) { rnd_seed[i] = (double) rand() / RAND_MAX; }

  cudaMemcpyToSymbol(dev_rnd_seed, rnd_seed, _RND_SEED_LENGTH * sizeof(double), 0, cudaMemcpyHostToDevice);

  dev_dense <double> *A = new dev_dense <double> (nx, ny), *B = new dev_dense <double> (nx, ny);

  B->loadTestMatrix(2000);

  timer myTimer = timer();

  myTimer.newEvent("qr", start);
  qr_kernel <<<1, 1024 >>> (A->getElements(), B->getElements(), nx, ny, nx, nx);
  myTimer.newEvent("qr", end);

  myTimer.dumpAllEvents_Sync();

  dev_dense <double> *m1 = A->matrixMultiplication(B), *m2 = new dev_dense<double>(nx, ny);
  m2->loadTestMatrix(2000);
  printf("Rel. L2 Error: %e\n\n", m2->L2Error(m1));
  dev_dense <double>* m3 = A->transpose()->matrixMultiplication(A), * m4 = new dev_dense<double>(nx, nx);
  m4->loadIdentityMatrix();
  printf("Rel. L2 Error: %e\n\n", m4->L2Error(m3));

  dev_dense <double>* m5 = A->matrixMultiplication(A->transpose()->matrixMultiplication(m2));
  printf("Rel. L2 Error: %e\n\n", m2->L2Error(m5));



  delete A; delete B; delete m1; delete m2; delete m3; delete m4; delete m5;


  return 0;
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

void test3()
{
  
}


int main(int argc, char **argv)
{
  //test0 <double> ();
  test1();
  //test2();

  return 0;
}