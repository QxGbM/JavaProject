
#include <dag.cuh>
#include <kernel.cuh>
#include <cuda_timer.cuh>

#if 0
using namespace cooperative_groups;

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
  struct dev_hierarchical <double> *a = new dev_hierarchical <double>(2, 2);
  a->loadTestMatrix(2, 2, 4);
  a->print();

  struct multi_level_index *id = new multi_level_index();
  struct ops_chain *ops = get_ops_h_getrf(a, id);
  ops->print();

  struct dag *d = new dag(ops);

  d->print();
  d->copyToDevice_Sync();

  struct timer *myTimer = new timer();
  myTimer->newEvent("TEST");

  void ** args = new void *[4]{ &d->length, &d->dev_dep, &d->dev_dep_counts, &d->dev_status };
  cudaLaunchKernel((void *)kernel_dynamic, 4, 256, args);
  delete[] args;

  myTimer->newEvent("TEST");
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
