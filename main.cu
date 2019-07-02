
#include <pspl.cuh>
#define ref


__global__ void partial_pivot_kernel(double *matrix, const int nx, const int ny, const int ld, int *pivot)
{
  __shared__ double shm[6144];
  blockDenseGetrf <double, 64, 48, 2> (matrix, nx, ny, ld, shm);
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

  const int blocks = 32, threads = 1024;
  cudaError_t error = hierarchical_GETRF <T, 12288> (a, blocks, threads);

#ifdef ref
  if (error == cudaSuccess)
  {
    dev_dense <T> * b = a->convertToDense(), * c = new dev_dense <T>(b->getNx(), b->getNy());

    stream = fopen("bin/ref.bin", "rb");
    c->loadBinary_ReverseEndian(stream);
    fclose(stream);

    timer my_timer = timer();
    my_timer.newEvent("ref", start);
    partial_pivot_kernel <<<1, 1024, 0, 0 >>> (c -> getElements(), c -> getNx(), c -> getNy(), c -> getLd(), nullptr);
    my_timer.newEvent("ref", end);

    my_timer.dumpAllEvents_Sync();

    printf("Rel. L2 Error: %e\n\n", c -> L2Error(b)); 
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

__global__ void ExampleKernel(int* d_data)
{

}

int test1()
{
  cudaSetDevice(0);
  cudaDeviceReset();


  srand(200);
  double * rnd_seed = new double[_RND_SEED_LENGTH];
#pragma omp parallel for
  for (int i = 0; i < _RND_SEED_LENGTH; i++) { rnd_seed[i] = (double) rand() / RAND_MAX; }

  cudaMemcpyToSymbol(dev_rnd_seed, rnd_seed, _RND_SEED_LENGTH * sizeof(double), 0, cudaMemcpyHostToDevice);

  int* data;
  cudaMallocManaged(&data, 512 * sizeof(int));
  for (int i = 0; i < 512; i++)
  { data[i] = i; }

  timer myTimer = timer();

  myTimer.newEvent("qr", start);
  ExampleKernel <<<1, 128 >>> (data);
  myTimer.newEvent("qr", end);

  myTimer.dumpAllEvents_Sync();


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
  //ref_mat->loadTestMatrix();

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