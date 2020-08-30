
#define RUN
#ifdef RUN

#include <matrix/Dense.cuh>
#include <matrix/Hierarchical.cuh>
#include <timer.cuh>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include <matrix/compressor.cuh>


int test0 (char test_name[], const int blocks, const int threads, const int kernel_size, const bool ref, char ref_name[]) {
  cudaSetDevice(0);
  cudaDeviceReset();

  int m = 3;
  int n = 3;
  Dense a = Dense(m, n);

  double test[]{1, 2, 3, 2, 5, 5, 3, 7, 3};
  a.load(&test[0], 3);

  auto arr = a.getElements();
  int ld = a.getLeadingDimension();
  a.print();

  std::cout << a.admissible(1);


  cusolverDnHandle_t handle;
  cusolverDnCreate(&handle);
  double* Workspace;
  int Lwork;
  cusolverDnDgetrf_bufferSize(handle, m, n, arr, ld, &Lwork);
  cudaMalloc(reinterpret_cast<void**>(&Workspace), Lwork);

  timer Timer = timer();

  Timer.newEvent("test");
  cusolverDnDgetrf(handle, m, n, arr, ld, Workspace, nullptr, nullptr);
  Timer.newEvent("test");


  Timer.dumpEvents();
  a.print();

  Hierarchical h = Hierarchical(16, 16, 2, 2);
  compressor c = compressor(h, 4, 0.5);

  return 0;
}



int main(int argc, char * argv[])
{
  int blocks = 80, threads = 512, kernel_size = 0, rank = 16;
  bool ref = false;

  char tmp[32], dir[32] = "bin/", ref_name[32], test_name[32] = "bin/test";

  for (int i = 1; i < argc; i++)
  {
    if (strncmp(argv[i], "-blocks=", 8) == 0)
    { sscanf(argv[i], "-blocks=%d", &blocks); }
    else if (strncmp(argv[i], "-threads=", 9) == 0)
    { sscanf(argv[i], "-threads=%d", &threads); }
    else if (strncmp(argv[i], "-kernel=", 8) == 0)
    { sscanf(argv[i], "-kernel=%d", &kernel_size); }
    else if (strncmp(argv[i], "-rank=", 6) == 0)
    { sscanf(argv[i], "-rank=%d", &rank); }
    else if (strncmp(argv[i], "-dir=", 5) == 0)
    { sscanf(argv[i], "-dir=%s", dir); }
    else if (strncmp(argv[i], "-test=", 6) == 0)
    { sscanf(argv[i], "-test=%s", tmp); strcpy(test_name, dir); strcat(test_name, tmp); }
    else if (strncmp(argv[i], "-ref=", 5) == 0)
    { sscanf(argv[i], "-ref=%s", tmp); strcpy(ref_name, dir); strcat(ref_name, tmp); ref = true; }
    else if (strcmp(argv[i], "-noref") == 0)
    { ref = false; }
    else
    { printf("Unrecognized Arg: %s.\n", argv[i]); }
  }

  test0 (test_name, blocks, threads, kernel_size, ref, ref_name);
  
  return 0;
}

#endif