
#include <pspl.cuh>

template <class T> __global__ void kernel(inst_handler <T> ih) { ih.run(); }

__host__ int test0()
{
  dev_hierarchical <double> *a = new dev_hierarchical <double>(2, 2);
  a->loadTestMatrix(1, 2, 4);
  a->print();

  const h_ops_tree *tree = a->generateOps_GETRF();
  tree->print();

  //dag *d = new dag(ops);
  //d->print();

  delete tree;
  delete a;

  cudaDeviceReset();

  return 0;
}

__host__ int test1 ()
{
  cudaSetDevice(0);

  dev_dense <float> *a = new dev_dense <float> (16, 16, 0, false);
  a -> loadTestMatrix();

  cudaStream_t main_stream;
  cudaStreamCreate(&main_stream);

  inst_handler <float> *ih = new inst_handler <float> (5);
  ih -> set_getrf_inst(0, a -> getElements(0), 8, 8, 16);
  ih -> set_trsml_inst(1, a -> getElements(0), a -> getElements(8), 8, 8, 8, 16, 16);
  ih -> set_trsmr_inst(2, a -> getElements(0), a -> getElements(128), 8, 8, 8, 16, 16);
  ih -> set_gemm_inst(3, a -> getElements(136), a -> getElements(128), a -> getElements(8), 8, 8, 8, 16, 16, 16);
  ih -> set_getrf_inst(4, a -> getElements(136), 8, 8, 16);

  ih -> add_dep(0, 1);
  ih -> add_dep(0, 2);
  ih -> add_dep(1, 3);
  ih -> add_dep(2, 3);
  ih -> add_dep(3, 4);

  ih -> fill_nop_inst();
  ih -> print();

  timer myTimer = timer();

  myTimer.newEvent("GETRF", start, main_stream);
  cudaLaunchKernel((void *)kernel <float>, 4, 1024, (void **) &ih, 0, main_stream);
  myTimer.newEvent("GETRF", end, main_stream);

  myTimer.printStatus();
  myTimer.dumpAllEvents_Sync();

  dev_dense <float> *b = a -> restoreLU();
  a -> loadTestMatrix();
  printf("Rel. L2 Error: %e\n\n", b -> L2Error(a));

  delete a;
  delete b;
  cudaStreamDestroy(main_stream);
  cudaDeviceReset();

  return 0;
}


int main(int argc, char **argv)
{
  test1();
  test0();

  return 0;
}