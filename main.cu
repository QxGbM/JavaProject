
#include <pspl.cuh>

__global__ void test_kernel(inst_handler <float> ih) { ih.run(); }

__host__ int test0()
{
  dev_hierarchical <double> *a = new dev_hierarchical <double>(2, 2);
  a->loadTestMatrix(2, 2, 4);
  a->print();

  multi_level_index *id = new multi_level_index();
  ops_chain *ops = get_ops_h_getrf(a, id);
  ops->print();

  dag *d = new dag(ops);

  d->print();

  timer *myTimer = new timer();
  myTimer->newEvent("TEST", start);

  void ** args = d -> getArgsAddress();
  cudaLaunchKernel((void *)kernel_dynamic, 4, 256, args);
  delete[] args;

  myTimer->newEvent("TEST", end);
  myTimer->printStatus();
  myTimer->dumpAllEvents_Sync();
  cudaDeviceSynchronize();

  delete a;
  delete ops;
  delete id;
  delete d;
  delete myTimer;

  cudaDeviceReset();

  return 0;
}

__host__ int test1 ()
{
  const int x = 4, y = 4;

  cudaSetDevice(0);

  dev_dense <float> *a = new dev_dense <float> (x, y);
  a -> loadTestMatrix();
  a -> print();

  dev_dense <float> *b = new dev_dense <float> (2 * x, y);
  b -> loadIdentityMatrix();
  b -> print();

  cudaStream_t main_stream;
  cudaStreamCreate(&main_stream);

  inst_handler <float> *ih = new inst_handler <float> (2);
  ih -> set_getrf_inst(0, a->getElements(), x, y, x);
  ih -> set_trsml_inst(1, a->getElements(), b->getElements(), x, y, 2 * x, x, 2 * x);
  ih -> add_dep(0, 1);
  ih -> print();

  timer myTimer = timer();

  myTimer.newEvent("GETRF", start, main_stream);
  cudaLaunchKernel((void *)test_kernel, 4, 1024, (void **) &ih, 0, main_stream);
  myTimer.newEvent("GETRF", end, main_stream);

  myTimer.printStatus();
  myTimer.dumpAllEvents_Sync();

  a->print();
  b->print();

  //a -> loadTestMatrix();

  //printf("Rel. L2 Error: %e\n\n", b -> L2Error(a));


  delete a;
  delete b;
  cudaStreamDestroy(main_stream);
  cudaDeviceReset();

  return 0;
}

__global__ void pivot_getrf_kernel(int *pivot, double *matrix, const int nx, const int ny, const int ld)
{
  blockDenseGetrf <double> (matrix, nx, ny, ld, pivot);
}

__global__ void recover_pivot_kernel(int *pivot, double *matrix, const int nx, const int ny, const int ld)
{
  blockApplyPivot <double> (matrix, pivot, nx, ny, ld, true);
}

__host__ int test2()
{
  cudaSetDevice(0);
  const int x = 512, y = 512;

  dev_dense <double> *a = new dev_dense <double>(x, y, 1024);
  a -> loadRandomMatrix(-10, 10, 999);

  int *dim = a -> getDim3(), nx = dim[0], ny = dim[1], ld = dim[2];
  double *matrix = a -> getElements();
  int *pivot = a -> getPivot();
  delete[] dim;

  timer myTimer = timer();
  void ** args = new void *[5]{ &pivot, &matrix, &nx, &ny, &ld };

  myTimer.newEvent("PIVOT GETRF", start);
  cudaLaunchKernel((void *)pivot_getrf_kernel, 1, 1024, args);
  myTimer.newEvent("PIVOT GETRF", end);
  cudaDeviceSynchronize();
  delete[] args;

  dev_dense <double> *b = a->restoreLU();
  double *matrix_b = b->getElements();
  void ** args2 = new void *[5]{ &pivot, &matrix_b, &nx, &ny, &nx };

  myTimer.newEvent("PIVOT", start);
  cudaLaunchKernel((void *)recover_pivot_kernel, 1, 1024, args2);
  myTimer.newEvent("PIVOT", end);

  myTimer.printStatus();
  myTimer.dumpAllEvents_Sync();
  delete[] args2;

  a->loadRandomMatrix(-10, 10, 999);
  printf("Rel. L2 Error: %e\n\n", b->L2Error(a));

  cudaDeviceReset();

  delete a;
  delete b;

  return 0;
}


int main(int argc, char **argv)
{
  //test2();
  test1();
  //test0();

  return 0;
}