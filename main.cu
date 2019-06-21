
#include <pspl.cuh>
#define ref


__global__ void partial_pivot_kernel(double *matrix, const int nx, const int ny, const int ld, int *pivot)
{
  __shared__ double shm[6144];
  blockDenseGetrf_shm <double>(matrix, pivot, nx, ny, ld, &shm[0]);
}

template <class T> __host__ int test0()
{
  cudaSetDevice(0);
  cudaDeviceReset();
  unsigned int rnd_seed_in = 200;

  T * rnd_seed = new T [_RND_SEED_LENGTH];
  srand(rnd_seed_in);

#pragma omp parallel for
  for (int i = 0; i < _RND_SEED_LENGTH; i++) 
  { rnd_seed[i] = (T) rand() / RAND_MAX; }

  cudaMemcpyToSymbol(dev_rnd_seed, rnd_seed, _RND_SEED_LENGTH * sizeof(T), 0, cudaMemcpyHostToDevice);
  delete[] rnd_seed;

  FILE * stream = fopen("bin/test.struct", "r");
  dev_hierarchical<T> * a = dev_hierarchical<T>::readStructureFromFile(stream);
  fclose(stream);

  stream = fopen("bin/test.bin", "rb");
  a -> loadBinary_ReverseEndian(stream);
  fclose(stream);

  const int blocks = 160, threads = 1024;
  cudaError_t error = hierarchical_GETRF <T, 12288> (a, blocks, threads);

#ifdef ref
  if (error == cudaSuccess)
  {
    dev_dense <T>* c = a->convertToDense(), * b = new dev_dense <T>(c->getNx(), c->getNy()); b -> loadTestMatrix();

    partial_pivot_kernel <<<1, 1024, 0, 0 >>> (b -> getElements(), b -> getNx(), b -> getNy(), b -> getLd(), nullptr);
    cudaDeviceSynchronize();

    printf("Rel. L2 Error: %e\n\n", b -> L2Error(c)); 
    delete b; b = nullptr;
    delete c; c = nullptr;
  }
#endif // ref
  delete a;

  return 0;
}


__global__ void qr_kernel (double* Q, double* R, const int nx, const int ny, const int ld_q, const int ld_r)
{
  blockGivensRotation <double> (R, nx, ny, ld_r);
  blockGivensRecoverQ <double> (Q, R, nx, ny, nx, ld_q, ld_r);

}

int test1()
{
  cudaSetDevice(0);
  cudaDeviceReset();

  const int nx = 8, ny = 8;

  srand(200);
  double * rnd_seed = new double[_RND_SEED_LENGTH];
#pragma omp parallel for
  for (int i = 0; i < _RND_SEED_LENGTH; i++) { rnd_seed[i] = (double) rand() / RAND_MAX; }

  cudaMemcpyToSymbol(dev_rnd_seed, rnd_seed, _RND_SEED_LENGTH * sizeof(double), 0, cudaMemcpyHostToDevice);

  dev_dense <double> *A = new dev_dense <double> (nx, ny), *B = new dev_dense <double> (nx, ny);

  B->loadTestMatrix(200); B->print();

  timer myTimer = timer();

  myTimer.newEvent("qr", start);
  qr_kernel <<<1, 1024 >>> (A->getElements(), B->getElements(), nx, ny, nx, nx);
  myTimer.newEvent("qr", end);

  myTimer.dumpAllEvents_Sync();
  A->print(); B->print();

  dev_dense <double> *m1 = A->matrixMultiplication(B), *m2 = new dev_dense<double>(nx, ny);
  m2->loadTestMatrix(200);
  printf("Rel. L2 Error: %e\n\n", m2->L2Error(m1));
  dev_dense <double>* m3 = A->transpose()->matrixMultiplication(A), * m4 = new dev_dense<double>(nx, nx);
  m4->loadIdentityMatrix();
  printf("Rel. L2 Error: %e\n\n", m4->L2Error(m3));

  dev_dense <double>* m5 = A->matrixMultiplication(A->transpose()->matrixMultiplication(m2));
  printf("Rel. L2 Error: %e\n\n", m2->L2Error(m5));

  delete A; delete B; delete m1; delete m2; delete m3; delete m4; delete m5;

  return 0;
}


void test2()
{
  FILE * stream = fopen("bin/test.struct", "r");
  dev_hierarchical<double> * h = dev_hierarchical<double>::readStructureFromFile(stream);
  fclose(stream);

  stream = fopen("bin/test.bin", "rb");
  h->loadBinary_ReverseEndian(stream);
  fclose(stream);
  //h->print();

  dev_dense<double> * d = h->convertToDense(), * ref_mat = new dev_dense<double> (d->getNx(), d->getNy());
  ref_mat->loadTestMatrix();

  printf("Rel. L2 Error: %e\n\n", d->L2Error(ref_mat));

  delete h;
  delete d;
}


int main(int argc, char **argv)
{
  test0 <double> ();
  //test1();
  //test2();

  return 0;
}