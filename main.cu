
#include <pspl.cuh>

template <class T> __global__ void kernel(inst_handler <T> ih) { ih.run(); }

__host__ int test0()
{
  const int n = 16, levels = 0, dim = 128, seed = 2000;

  dev_hierarchical <double> *a = new dev_hierarchical <double> (n, n);
  a -> loadTestMatrix(levels, n, dim, seed);
  printf("Testing: %d x %d.\n", a -> getNy(), a -> getNx());

  h_ops_dag *d = new h_ops_dag(a -> generateOps_GETRF());

  inst_handler <double> *ih = new inst_handler <double> (d, a);

  timer myTimer = timer();
  cudaStream_t main_stream;
  cudaStreamCreate(&main_stream);

  myTimer.newEvent("GETRF", start, main_stream);
  cudaLaunchKernel((void *)kernel <double>, 64, 1024, (void **)&ih, 0, main_stream);
  myTimer.newEvent("GETRF", end, main_stream);

  myTimer.printStatus();
  myTimer.dumpAllEvents_Sync();

  dev_dense <double> *b = a -> convertToDense() -> restoreLU();

  a -> loadTestMatrix(levels, n, dim, seed);
  dev_dense <double> *c = a -> convertToDense();

  printf("Rel. L2 Error: %e\n\n", b -> L2Error(c));

  delete ih, d, a;
  delete b, c;

  cudaStreamDestroy(main_stream);
  cudaDeviceReset();

  return 0;
}


int main(int argc, char **argv)
{
  test0();

  return 0;
}