
#ifndef _KERNEL_CUH
#define _KERNEL_CUH

#include <pspl.cuh>

template <class T, int shm_size> 
__global__ void kernel_dynamic (const int ** __restrict__ insts, void ** __restrict__ ptrs, volatile int * __restrict__ comm_space)
{
  __shared__ int shm [shm_size]; 

  const int * pc = insts [block_rank()], shm_size_acutal = shm_size * 4 / sizeof(T);

load_inst:
  if (thread_rank() < _MAX_INST_LENGTH)
  { shm[thread_rank()] = pc[thread_rank()]; }
  int next_pc = 0;
  __syncthreads();

  switch ((opcode_t) shm[0])
  {
  case execute: goto exe;
  case signal_wait: goto wait;
  case signal_write: goto write;
  case finish: goto fin;

  default: goto fin;
  }

exe:
  switch ((operation_t) shm[1])
  {
  case getrf_d:
  {
    T * M = (T *) ptrs[shm[2]]; 
    const int offset = shm[3], ld = shm[4], nx = shm[5], ny = shm[6];
    __syncthreads();
    blockDenseGetrf_shm <T> (&M[offset], nullptr, nx, ny, ld, (T *) shm);
    next_pc = 7; goto sync;  
  }

  case trsml_d:
  {
    T * B = (T *) ptrs[shm[2]], * L = (T *) ptrs[shm[5]];
    const int offset_b = shm[3], ld_b = shm[4], offset_l = shm[6], ld_l = shm[7], nx_b = shm[8], ny_b = shm[9], nx_l = shm[10];
    __syncthreads();
    blockDenseTrsmL_shm <T> (&B[offset_b], &L[offset_l], nx_b, ny_b, nx_l, ld_b, ld_l, false, (T *) shm, shm_size_acutal);
    next_pc = 11; goto sync;  
  }

  case trsmr_d:
  {
    T * B = (T *) ptrs[shm[2]], * U = (T *) ptrs[shm[5]];
    const int offset_b = shm[3], ld_b = shm[4], offset_u = shm[6], ld_u = shm[7], nx_b = shm[8], ny_b = shm[9], ny_u = shm[10];
    __syncthreads();
    blockDenseTrsmR_shm <T> (&B[offset_b], &U[offset_u], nx_b, ny_b, ny_u, ld_b, ld_u, false, (T *) shm, shm_size_acutal);
    next_pc = 11; goto sync;  
  }

  case gemm_d_d_d:
  {
    T * M = (T *) ptrs[shm[2]], * A = (T *) ptrs[shm[5]], * B = (T *) ptrs[shm[8]];
    const int offset_m = shm[3], ld_m = shm[4], offset_a = shm[6], ld_a = shm[7], offset_b = shm[9], ld_b = shm[10], m = shm[11], n = shm[12], k = shm[13];
    __syncthreads();
    blockDenseGemm_shm <T> (-1., 1., &M[offset_m], &A[offset_a], &B[offset_b], m, n, k, ld_m, ld_a, ld_b, false, false, (T *) shm, shm_size_acutal);
    next_pc = 14; goto sync;
  }

  default: goto fin;
  }

wait:
  if (thread_rank() == 0)
  { shm[0] = comm_space[shm[1]]; }
  __syncthreads();
  if (shm[0])
  { next_pc = 2; }
  goto sync;

write:
  if (thread_rank() == 0)
  { comm_space[shm[1]] = 1; }
  __threadfence();
  next_pc = 2;
  goto sync;

sync:
  __syncthreads();
  if (next_pc > 0) 
  { pc = &pc[next_pc]; goto load_inst; }
  else
  { goto wait; }
  
fin:
  return;
}

template <class T, int shm_size> 
__host__ cudaError_t hierarchical_GETRF (dev_hierarchical <T> * h, const int num_blocks, const int num_threads)
{
  cudaSetDevice(0);
  if (sizeof(T) == 8 && cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte) == cudaSuccess)
  { printf("Shared memory bank size configured to be 8-bytes.\n"); }

  cudaDeviceProp deviceprop;
  cudaGetDeviceProperties(&deviceprop, 0);
  int numSMs = deviceprop.multiProcessorCount, numBlocksPerSm = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, (void *) kernel_dynamic <T, shm_size>, num_threads, 0);
  printf("# SMs: %d, # Blocks per SM for launch: %d\n\n", numSMs, numBlocksPerSm);

  const int workers_max = numSMs * numBlocksPerSm, workers = workers_max < num_blocks ? workers_max : num_blocks;
  if (workers == 0)
  { printf("Launch Config: Too many resources requested for launch.\n\n"); return cudaErrorInvalidConfiguration; }
  else if (workers < num_blocks)
  { printf("Launch Config: Number of launched blocks reduced from %d to %d. \n\n", num_blocks, workers); }

  const int ny = h -> getNy_abs(), nx = h -> getNx_abs();
  printf("Start Testing Hierarchical - LU for: %d x %d.\n\n", ny, nx);

  timer myTimer = timer();
  cudaStream_t main_stream;
  cudaStreamCreate(&main_stream);

  double clock_start, clock_end;

  clock_start = omp_get_wtime();
  if (!(h -> partitionForLU()))
  { printf("-- Partition Failed. Aborting. --\n"); return cudaErrorUnknown; }
  clock_end = omp_get_wtime();
  printf("LU partition finishes in %f ms.\n\n", 1000. * (clock_end - clock_start));

  clock_start = omp_get_wtime();
  const h_index * root = h -> getRootIndex();
  const h_ops_tree * tree = h -> generateOps_GETRF(root);
  clock_end = omp_get_wtime();
  printf("Tree Generated in %f ms.\n\n", 1000. * (clock_end - clock_start));

  /*clock_start = omp_get_wtime();
  h_ops_dag dag = h_ops_dag (tree);
  clock_end = omp_get_wtime();
  delete tree;
  printf("DAG Created in %f ms.\n\n", 1000. * (clock_end - clock_start));

  clock_start = omp_get_wtime();
  instructions_scheduler schedule = instructions_scheduler (&dag, workers);
  clock_end = omp_get_wtime();
  printf("Schedule Created in %f ms.\n\n", 1000. * (clock_end - clock_start));

  clock_start = omp_get_wtime();
  instructions_manager ins = instructions_manager (workers, &dag, &schedule);
  clock_end = omp_get_wtime();
  printf("Instruction generated in %f ms.\n\n", 1000. * (clock_end - clock_start));

  int ** dev_insts, * comm_space;
  void ** args, ** dev_ptrs;
  fprintf(stderr, "Args: %s\n\n", cudaGetErrorString(ins.getLaunchArgs(&dev_insts, &dev_ptrs, &comm_space)));
  args = new void *[3] { &dev_insts, &dev_ptrs, &comm_space };

  myTimer.newEvent("Kernel", start, main_stream);
  cudaError_t error = cudaLaunchKernel((void *) kernel_dynamic <T, shm_size>, workers, num_threads, args, 0, main_stream);
  myTimer.newEvent("Kernel", end, main_stream);

  fprintf(stderr, "Kernel Launch: %s\n\n", cudaGetErrorString(error));

  h_ops dense_op = h_ops (getrf_d, root);
  delete root;
  delete args;

  const unsigned long long int exeFLOPS = dag.getFops(), estFLOPS = dense_op.getFops();
  const double exeTime = myTimer.dumpAllEvents_Sync(), compressRatio = 100. * exeFLOPS / estFLOPS;

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

  gpuflops *= 100. / compressRatio;

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

  cudaStreamDestroy(main_stream);*/

  return cudaSuccess;
}



#endif