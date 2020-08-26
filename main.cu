
#define RUN
#ifdef RUN

#include <matrix/Dense.cuh>
#include <matrix/Hierarchical.cuh>
#include <launcher.cuh>
#include <timer.cuh>
#include <cusolverDn.h>


int test0 (char test_name[], const int blocks, const int threads, const int kernel_size, const bool ref, char ref_name[])
{
  cudaSetDevice(0);
  cudaDeviceReset();

  Hierarchical * a = Hierarchical :: readFromFile(test_name, 0);
  //a->print();

  cudaError_t error = dev_hierarchical_GETRF(a, blocks, threads, kernel_size);

  if (ref && error == cudaSuccess)
  {
    Dense * b = a->convertToDense(), * c = Dense :: readFromFile(ref_name, 0);
    b->print();
    c->print();

    int m = c->getNy();
    int n = c->getNx();
    int ld = c->getLd();
    auto arr = c->getElements();

    cusolverDnHandle_t handle;
    cusolverDnCreate(&handle);
    double* Workspace;
    int Lwork;
    int* devInfo;
    cusolverDnDgetrf_bufferSize(handle, m, n, arr, ld, &Lwork);
    Lwork = 16384;
    cudaMalloc(&Workspace, Lwork);
    cudaMalloc(&devInfo, sizeof(int));

    timer my_timer = timer();
    my_timer.newEvent("ref", start);
    cusolverDnDgetrf(handle, m, n, arr, ld, Workspace, nullptr, devInfo);
    my_timer.newEvent("ref", end);


    my_timer.dumpAllEvents_Sync();

    printf("\033[0;31m");
    printf("Rel. L2 Error: %e\n\n", c -> L2Error(b)); 
    printf("\033[0m");

    delete b; b = nullptr;
    delete c; c = nullptr;
  }

  delete a;

  return 0;
}



int main(int argc, char * argv[])
{
  int blocks = 80, threads = 512, kernel_size = 0, rank = _SHADOW_RANK;
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