
#include <stdio.h>
#include <cuda.h>
#include <cooperative_groups.h>

#include <dev_dense.cuh>
#include <cuda_timer.cuh>

using namespace cooperative_groups;

struct dev_hierarchical {

  int nx;
  int ny;
  void **elements;
  int *elements_type;
  
  dev_hierarchical(const int x, const int y)
  {
    nx = x;
    ny = y;
    elements = (void **) malloc (x * y * sizeof(void *));
    elements_type = (int *) malloc (x * y * sizeof(int *));

    for (int i = 0; i < x * y; i++)
    { elements[i] = nullptr; elements_type[i] = 0; }
  }

  ~dev_hierarchical()
  {
    for (int i = 0; i < nx * ny; i++)
    { if (elements[i] != nullptr) free(elements[i]); }
    free(elements_type);
  }

  int set_element(struct dev_hierarchical *matrix, int x, int y) 
  {
    elements[y * nx + x] = (void *) matrix;
    elements_type[y * nx + x] = 3;
    return 0;
  }

  int set_element(struct dev_dense <double> *matrix, int x, int y)
  {
    elements[y * nx + x] = (void *) matrix;
    elements_type[y * nx + x] = 1;
    return 0;
  }

  void print()
  {
    printf("%d, %d\n", nx, ny);
    for (int i = 0; i < nx * ny; i++)
    { printf("%d, %d\n", elements[i] == nullptr, elements_type[i]); }
  }

};

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
  test_kernel();

  /*struct dev_dense <double> a = dev_dense <double> (3, 3);
  a.loadRandomMatrix(-10, 10, 888);
  struct dev_dense <double> b = dev_dense <double> (3, 3);
  b.loadTestMatrix();
  struct dev_dense <double> *c = b.restoreLU();
  a.print();
  a.copyToDevice_Sync();
  b.print();
  c -> print();*/
  return 0;
}