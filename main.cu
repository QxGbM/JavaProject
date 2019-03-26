
#include <pspl.cuh>

template <class T> __global__ void kernel(inst_handler <T> ih) { ih.run(); }

__host__ int test0()
{
  dev_hierarchical <double> *a = new dev_hierarchical <double> (2, 2);
  a -> loadTestMatrix(2, 2, 4, 999);

  h_ops_dag *d = new h_ops_dag(a -> generateOps_GETRF());

  inst_handler <double> *ih = new inst_handler <double> (d, a);

  timer myTimer = timer();
  cudaStream_t main_stream;
  cudaStreamCreate(&main_stream);

  myTimer.newEvent("GETRF", start, main_stream);
  cudaLaunchKernel((void *)kernel <double>, 8, 1024, (void **)&ih, 0, main_stream);
  myTimer.newEvent("GETRF", end, main_stream);

  myTimer.printStatus();
  myTimer.dumpAllEvents_Sync();

  dev_dense <double> *b = a -> convertToDense() -> restoreLU();

  a -> loadTestMatrix(2, 2, 4, 999);
  dev_dense <double> *c = a -> convertToDense();

  printf("Rel. L2 Error: %e\n\n", b->L2Error(c));

  delete ih;
  delete d;
  delete a;
  delete b;
  delete c;

  cudaStreamDestroy(main_stream);
  cudaDeviceReset();

  return 0;
}


int main(int argc, char **argv)
{
  test0();

  return 0;
}