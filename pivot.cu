
#include <pivot.cuh>
#include <dense_getrf.cuh>
#include <dev_dense.cuh>
#include <cuda_timer.cuh>

__global__ void partial_pivot_kernel (int *pivot, double *matrix, const int nx, const int ld, const int ny)
{
  blockDenseGetrfWithPivot <double, 32> (pivot, matrix, nx, ld, ny);
}

__global__ void recover_pivot_kernel (int *pivot, double *matrix, const int nx, const int ld, const int ny)
{
  blockApplyPivot <double> (this_thread_block(), pivot, true, matrix, nx, ld, ny);
}

__host__ int main()
{
  cudaSetDevice(0);
  const int nx = 16, ld = 16, ny = 16;

  struct dev_dense <double> *a = new dev_dense <double> (nx, ny, ld);
  a -> loadRandomMatrix(-10, 10, 999);
  a -> print();
  a -> copyToDevice_Sync();

  struct timer myTimer = timer();

  dim3 block(128), grid(1);
  myTimer.newEvent("pivot");
  partial_pivot_kernel <<<grid, block, 0, 0>>> (a -> dev_pivot, a -> dev_ptr, nx, ld, ny);
  myTimer.newEvent("pivot");
  cudaDeviceSynchronize();

  a -> copyToHost_Sync();
  a -> print();

  struct dev_dense <double> *b = a -> restoreLU();
  b -> print();
  b -> copyToDevice_Sync();

  myTimer.newEvent("pivot recovery");
  recover_pivot_kernel <<<grid, block, 0, 0>>> (a -> dev_pivot, b -> dev_ptr, nx, ld, ny);
  myTimer.newEvent("pivot recovery");

  myTimer.printStatus();
  myTimer.dumpAllEvents_Sync();

  b -> copyToHost_Sync();
  b -> print();

  cudaDeviceReset();
  return 0;
}