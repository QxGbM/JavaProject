

#include <definitions.cuh>
#include <launcher.cuh>

#include <timer.cuh>
#include <h_ops/dev_hierarchical_index.cuh>
#include <h_ops/dev_hierarchical_ops_dag.cuh>
#include <h_ops/dev_hierarchical_ops_tree.cuh>
#include <h_ops/dev_hierarchical_ops.cuh>
#include <dev_temp.cuh>
#include <matrix/Hierarchical.cuh>
#include <instructions/instructions_scheduler.cuh>
#include <instructions/instructions_manager.cuh>

#include <kernel_cublas.cuh>

void print_dev_mat (real_t * dev_mat, const int nx, const int ny)
{
   real_t * data = new real_t [(size_t) nx * ny];
   cudaMemcpy (data, dev_mat, (size_t) nx * ny * sizeof(real_t), cudaMemcpyDeviceToHost);
   for (int i = 0; i < ny; i++)
   {
     for (int j = 0; j < nx; j++)
     { printf("%e ", data[i * nx + j]); }
     printf("\n");
   }
   delete[] data;
}

cudaError_t allocate_clocks (unsigned long long *** clocks, const int workers, const int * lengths)
{
  unsigned long long ** tmp = new unsigned long long * [workers];
  cudaMalloc(clocks, workers * sizeof(unsigned long long *));

  for (int i = 0; i < workers; i++)
  {
    cudaMalloc(&tmp[i], ((size_t) 1 + lengths[i]) * sizeof(unsigned long long));
    cudaMemset(tmp[i], 0, ((size_t)1 + lengths[i]) * sizeof(unsigned long long));
  }
  cudaMemcpy(* clocks, tmp, workers * sizeof(unsigned long long *), cudaMemcpyHostToDevice);

  return cudaGetLastError();
}

cudaError_t generateLaunchArgsFromTree (int *** dev_insts, void *** dev_ptrs, int ** comm_space, real_t *** block_tmps, real_t ** dev_rnd_seed, unsigned long long *** clocks,
  instructions_scheduler ** schedule_addr, double * total_lapse, long long * flops, const h_ops_tree * tree, real_t ** tmp_ptrs, const int workers, const int start_index, const int length_max)
{
  double clock_start, clock_end, clock_lapse, clock_total = 0.;
  printf("-- Host Summary: -- \n");

  clock_start = omp_get_wtime();
  h_ops_dag dag = h_ops_dag (tree, start_index, length_max);
  clock_end = omp_get_wtime();
  clock_lapse = clock_end - clock_start;
  clock_total += clock_lapse;
  printf("DAG Created in %f ms.\n", 1000. * clock_lapse); //dag.print(); 
  std::cout << "Density: " << dag.dag_density() << std::endl;

  clock_start = omp_get_wtime();
  * schedule_addr = new instructions_scheduler (&dag, workers);
  clock_end = omp_get_wtime();
  clock_lapse = clock_end - clock_start;
  clock_total += clock_lapse;
  printf("Schedule Created in %f ms.\n", 1000. * clock_lapse); //schedule.print();

  int * lengths = (* schedule_addr) -> getLengths();
  allocate_clocks(clocks, workers, lengths);
  delete lengths;

  clock_start = omp_get_wtime();
  instructions_manager ins = instructions_manager (workers, &dag, * schedule_addr, (void **) tmp_ptrs);
  clock_end = omp_get_wtime();
  clock_lapse = clock_end - clock_start;
  clock_total += clock_lapse;
  printf("Instruction generated in %f ms.\n", 1000. * clock_lapse); //ins.print();

  int n_streams, n_insts;
  using std::vector;
  vector<int>* insts;
  vector<double*> ptrs;
  ins.getLaunchArgsCublas(n_streams, n_insts, insts, ptrs);
  printf("Args generated in %f ms.\n", 1000. * clock_lapse);

  kernel_cublas(n_streams, n_insts, insts, ptrs);

  * total_lapse = clock_total;
  * flops = dag.getFlops();
  return cudaSuccess;
}

cudaError_t launchKernelWithArgs (int ** dev_insts, void ** dev_ptrs, int * comm_space, real_t ** block_tmps, real_t * dev_rnd_seed, unsigned long long ** clocks, 
  const int workers, const int num_threads, cudaStream_t main_stream)
{
  void ** args = new void * [6] { &dev_insts, &dev_ptrs, &comm_space, &block_tmps, &dev_rnd_seed, &clocks };
  cudaError_t error = cudaSuccess;// cudaLaunchKernel((void*)kernel_dynamic, workers, num_threads, args, 0, main_stream);
  fprintf(stderr, "Kernel Launch: %s\n\n", cudaGetErrorString(error));

  /*cudaDeviceSynchronize();
  for (int i = 0; i < workers; i++)
  {
    cudaFree(dev_insts[i]); // creates seg fault due to dev_insts is on device;
    if (block_tmps[i] != nullptr)
    { cudaFree(block_tmps[i]); }
  }*/

  cudaFree(dev_insts);
  cudaFree(dev_ptrs);
  cudaFree(comm_space);
  cudaFree(block_tmps);
  cudaFree(dev_rnd_seed);
  delete[] args;

  return error;
}

cudaError_t dev_hierarchical_GETRF (Hierarchical * h, const int num_blocks, const int num_threads, const int kernel_size)
{
  cudaSetDevice(0);
  if (sizeof(real_t) == 8 && cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte) == cudaSuccess)
  { printf("Shared memory bank size configured to be 8-bytes.\n"); }

  int numSMs = 1;
  printf("# SMs: %d\n\n", numSMs);

  const int ny = h -> getNy_abs(), nx = h -> getNx_abs();
  printf("Start Testing Hierarchical - LU for: %d x %d.\n\n", ny, nx);

  timer myTimer = timer();
  cudaStream_t main_stream;
  cudaStreamCreate(&main_stream);

  double clock_start, clock_end, clock_lapse;
  cudaError_t error = cudaSuccess;
  dev_temp tmp_mngr = dev_temp();

  clock_start = omp_get_wtime();
  const h_index * root = h -> getRootIndex();
  const h_ops_tree * tree = h -> generateOps_GETRF(root, &tmp_mngr);
  clock_end = omp_get_wtime();
  clock_lapse = clock_end - clock_start;
  printf("Tree Generated in %f ms.\n\n", 1000. * clock_lapse); //tree->print();
  delete root;

  real_t ** tmp_ptrs = tmp_mngr.allocate(), ** block_tmps, * dev_rnd_seed;
  int ** dev_insts, * comm_space, iters = kernel_size <= 0 ? 1 : (tree -> length() + kernel_size - 1) / kernel_size;
  void ** dev_ptrs;
  long long int exeFLOPS = 0, tmp;
  unsigned long long int ** clocks;
  char event_name[32];

  for (int i = 0; i < iters && error == cudaSuccess; i++)
  {
    instructions_scheduler * schedule;
    error = generateLaunchArgsFromTree (&dev_insts, &dev_ptrs, &comm_space, &block_tmps, &dev_rnd_seed, &clocks, &schedule, &clock_lapse, &tmp, tree, tmp_ptrs, numSMs, i * kernel_size, kernel_size);
    printf("Host %f ms.\n\n", 1000. * clock_lapse);
    exeFLOPS += tmp;

    sprintf(event_name, "Kernel %d", i);

    /*myTimer.newEvent(event_name, start, main_stream);
    error = launchKernelWithArgs (dev_insts, dev_ptrs, comm_space, block_tmps, dev_rnd_seed, clocks, numSMs, num_threads, main_stream);
    myTimer.newEvent(event_name, end, main_stream);*/

    //schedule -> analyzeClocks(clocks);
    delete schedule;
  }

  const double exeTime = myTimer.dumpAllEvents_Sync();

  cudaFree(tmp_ptrs[0]);
  delete[] tmp_ptrs;

  const long long int estFLOPS = h_ops::getFlops_GETRF(&tmp, nx, ny);
  const double compressRatio = estFLOPS == 0 ? 0 : 100. * exeFLOPS / estFLOPS;

  printf("-- Kernel Running Summary --\n"
    "Actual FLOPS: %llu.\nDense-LU FLOPS: %llu.\nFLOPS Compression Ratio: %f%%.\n", 
    exeFLOPS, estFLOPS, compressRatio);

  double gpuflops = 1.e3 * exeFLOPS / exeTime;
  int power = 0;

  while (power < 4 && gpuflops > 1.e3) 
  { gpuflops *= 1.e-3; power ++; }
  printf("GPU: %f ", gpuflops);

  switch (power)
  {
  case 0: break;
  case 1: printf("K"); break;
  case 2: printf("M"); break;
  case 3: printf("G"); break;
  case 4: printf("T"); break;
  }
  printf("FLOPS/S.\n");

  gpuflops *= compressRatio == 0 ? 0 : 100. / compressRatio;

  while (power < 4 && gpuflops > 1.e3) 
  { gpuflops *= 1.e-3; power ++; }
  printf("Equivalent Dense-LU: %f ", gpuflops);

  switch (power)
  {
  case 0: break;
  case 1: printf("K"); break;
  case 2: printf("M"); break;
  case 3: printf("G"); break;
  case 4: printf("T"); break;
  }
  printf("FLOPS/S.\n\n");

  error = cudaStreamDestroy(main_stream);

  return error;
}

