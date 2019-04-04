
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


int main(int argc, char **argv)
{
  test0 <double> ();
  //test1();

  return 0;
}