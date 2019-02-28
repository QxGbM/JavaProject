
#include <stdio.h>
#include <cuda.h>
#include <cooperative_groups.h>

#include <dev_dense.cuh>
#include <cuda_timer.cuh>
#include <dev_hierarchical.cuh>
#include <dag.cuh>

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

__host__ int main()
{
  //test_kernel();

  struct dev_hierarchical <double> a = dev_hierarchical <double> (2, 2);
  a.loadTestMatrix(1, 2, 4);
  //a.print();

  //int n[] = {0, 3};
  //struct multi_level_index i = multi_level_index(2, &n[0]);
  //struct multi_level_index i2 = multi_level_index(2, &n[0]);

  //printf("%d\n", i2.compare(&i));

  //struct h_matrix_element <double> *e = a.lookup(&l);
  //e -> print();

  //struct ops_chain *c = get_ops_hgetrf(&a);
  //c -> print();

  struct dag d = dag(get_ops_hgetrf(&a));
  //struct ops_chain *my_c = c -> lookup(10);
  //my_c -> print(0, true, false);
  d.print();

  return 0;
}