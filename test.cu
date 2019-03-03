
#include <kernel.cuh>
#include <cuda_timer.cuh>

using namespace cooperative_groups;

#if 0
__global__ void kernel (int *in)
{
  grid_group grid = this_grid();
  for (int i = 0; i < 10; i++)
  {
    atomicAdd(in, 1);
    grid.sync();
    if (grid.thread_rank() == 0) { printf("%d: %d\n", i, *in); }
    grid.sync();
  }
}

__host__ void test_kernel()
{
  
  struct arguments {
    int *in;
  };

  int numBlocksPerSm = 0, numThreads = 64;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, kernel, numThreads, 0);
  printf("#threads: %d, #blocks: %d\n", numThreads, numBlocksPerSm);

  struct arguments *args = (struct arguments*) malloc(sizeof(struct arguments));
  int *dev_num = 0, num = 0;
  cudaMalloc((void**) &dev_num, sizeof(int));

  cudaMemcpy(&dev_num, &num, sizeof(int), cudaMemcpyHostToDevice);
  args -> in = dev_num;

  struct timer myTimer = timer();
  myTimer.newEvent("test");
  myTimer.newEvent("test2");
  cudaLaunchCooperativeKernel((void *)kernel, 1 * numBlocksPerSm, numThreads, (void **) &args);
  myTimer.newEvent("test2");

  cudaLaunchCooperativeKernel((void *)kernel, 1 * numBlocksPerSm, numThreads, (void **) &args);

  myTimer.newEvent("test2");
  cudaLaunchCooperativeKernel((void *)kernel, 1 * numBlocksPerSm, numThreads, (void **) &args);
  myTimer.newEvent("test2");

  myTimer.newEvent("test");

  myTimer.printStatus();
  myTimer.dumpAllEvents_Sync();
}

#endif

__host__ int main()
{
  //test_kernel();

  struct dev_hierarchical <double> *a = new dev_hierarchical <double> (2, 2);
  a -> loadTestMatrix(1, 2, 4);
  a -> print();

  struct dag *d = new dag(get_ops_hgetrf(a));

  d -> print();
  d -> copyToDevice_Sync();

  struct timer *myTimer = new timer();
  myTimer -> newEvent("TEST");

  kernel_dynamic <<<4, 256>>> (d -> length, d -> dev_dep, d -> dev_dep_counts, d -> dev_status);

  myTimer -> newEvent("TEST");
  myTimer -> printStatus();
  myTimer -> dumpAllEvents_Sync();
  cudaDeviceSynchronize();

  delete a;
  delete d;
  delete myTimer;

  return 0;
}