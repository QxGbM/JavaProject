
#include <pspl.cuh>
#define ref


__global__ void partial_pivot_kernel(double *matrix, const int nx, const int ny, const int ld, int *pivot)
{
  __shared__ double shm[6144];
  blockDenseGetrf <double, double2, 2, _DEFAULT_BLOCK_M, _DEFAULT_BLOCK_K> (matrix, nx, ny, ld, shm);
  //DenseGetrf <double, double2, 2> (matrix, nx, ny, ld);
}

template <class T, class vecT, int vec_size> __host__ int test0()
{
  cudaSetDevice(0);
  cudaDeviceReset();

  rndInitialize <T> (200);

  FILE * stream = fopen("bin/test.struct", "r");
  dev_hierarchical<T> * a = dev_hierarchical<T>::readStructureFromFile(stream);
  fclose(stream);

  stream = fopen("bin/test.bin", "rb");
  a -> loadBinary(stream);
  fclose(stream);

  const int blocks = 68, threads = 512;
  cudaError_t error = hierarchical_GETRF <T, vecT, vec_size, 12288> (a, blocks, threads);

#ifdef ref
  if (error == cudaSuccess)
  {
    dev_dense <T> * b = a->convertToDense(), * c = new dev_dense <T>(b->getNx(), b->getNy());

    stream = fopen("bin/ref.bin", "rb");
    c->loadBinary(stream);
    fclose(stream);

    timer my_timer = timer();
    my_timer.newEvent("ref", start);
    partial_pivot_kernel <<<1, threads, 0, 0 >>> (c -> getElements(), c -> getNx(), c -> getNy(), c -> getLd(), nullptr);
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



int main(int argc, char **argv)
{
  test0 <double, double2, 2> ();

  return 0;
}