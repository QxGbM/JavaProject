
#ifndef _KERNEL_CUH
#define _KERNEL_CUH

#include <pspl.cuh>

template <class T, int shm_size> __global__ void __launch_bounds__ (1024, 2)
  kernel_dynamic (const int ** __restrict__ insts, T ** __restrict__ ptrs, int ** __restrict__ pivot_ptrs, volatile int * __restrict__ comm_space)
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
    T * M = ptrs[shm[2]]; 
    int * p = (shm[3] == -1) ? nullptr : (int *) pivot_ptrs[shm[3]], nx = shm[4], ny = shm[5], ld = shm[6];
    __syncthreads();
    blockDenseGetrf_shm <T> (M, p, nx, ny, ld, (T *) shm);
    next_pc = 7; goto sync;  
  }

  case trsml_d:
  {
    T * B = ptrs[shm[2]], * L = ptrs[shm[3]];
    int nx_b = shm[4], ny_b = shm[5], nx_l = shm[6], ld_b = shm[7], ld_l = shm[8];
    __syncthreads();
    blockDenseTrsmL_shm <T> (B, L, nx_b, ny_b, nx_l, ld_b, ld_l, false, (T *) shm, shm_size_acutal);
    next_pc = 9; goto sync;  
  }

  case trsml_lr:
  {
    T * B = ptrs[shm[2]], * L = ptrs[shm[3]];
    int nx_b = shm[4], ny_b = shm[5], nx_l = shm[6], ld_b = shm[7], ld_l = shm[8];
    bool b_T = (bool) shm[9];
    __syncthreads();
    blockDenseTrsmL_shm <T> (B, L, nx_b, ny_b, nx_l, ld_b, ld_l, b_T, (T *) shm, shm_size_acutal);
    next_pc = 10; goto sync;
  }

  case trsmr_d:
  {
    T * B = ptrs[shm[2]], * U = ptrs[shm[3]];
    int nx_b = shm[4], ny_b = shm[5], ny_u = shm[6], ld_b = shm[7], ld_u = shm[8];
    __syncthreads();
    blockDenseTrsmR_shm <T> (B, U, nx_b, ny_b, ny_u, ld_b, ld_u, false, (T *) shm, shm_size_acutal);
    next_pc = 9; goto sync;  
  }

  case trsmr_lr:
  {
    T * B = ptrs[shm[2]], * U = ptrs[shm[3]];
    int nx_b = shm[4], ny_b = shm[5], ny_u = shm[6], ld_b = shm[7], ld_u = shm[8];
    bool b_T = (bool) shm[9];
    __syncthreads();
    blockDenseTrsmR_shm <T> (B, U, nx_b, ny_b, ny_u, ld_b, ld_u, b_T, (T *) shm, shm_size_acutal);
    next_pc = 10; goto sync;  
  }

  case gemm_d_d_d:
  {
    T * M = ptrs[shm[2]], * A = ptrs[shm[3]], * B = ptrs[shm[4]];
    int m = shm[5], n = shm[6], k = shm[7], ld_m = shm[8], ld_a = shm[9], ld_b = shm[10];
    bool a_T = (bool) shm[11], b_T = (bool) shm[12];
    __syncthreads();
    blockDenseGemm_shm <T> (-1., 1., M, A, B, m, n, k, ld_m, ld_a, ld_b, a_T, b_T, (T *)shm, shm_size_acutal);
    next_pc = 13; goto sync;
  }

  case pivot_d:
  {
    T * M = ptrs[shm[2]];
    int * p = pivot_ptrs[shm[3]], nx = shm[4], ny = shm[5], ld = shm[6];
    bool p_T = (bool) shm[7];
    __syncthreads();
    blockApplyPivot <T> (M, p, nx, ny, ld, p_T, (T *) shm, shm_size_acutal);
    next_pc = 8; goto sync;
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

  const int ny = h -> getNy(), nx = h -> getNx();
  printf("Start Testing Hierarchical - LU for: %d x %d.\n\n", ny, nx);

  timer myTimer = timer();
  cudaStream_t main_stream;
  cudaStreamCreate(&main_stream);

  double clock_start, clock_end;

  const h_index * root = h -> getRootIndex();
  clock_start = omp_get_wtime();
  const h_ops_tree * tree = h -> generateOps_GETRF(root);
  clock_end = omp_get_wtime();
  printf("Tree Generated in %f ms.\n\n", 1000. * (clock_end - clock_start));

  tree->print();
  delete tree;
  delete root;

  /*clock_start = omp_get_wtime();
  h_ops_dag dag = h_ops_dag (tree);
  clock_end = omp_get_wtime();
  delete tree;
  printf("DAG Created in %f ms.\n\n", 1000. * (clock_end - clock_start));

  clock_start = omp_get_wtime();
  inst_scheduler schedule = inst_scheduler (&dag, workers);
  clock_end = omp_get_wtime();
  printf("Schedule Created in %f ms.\n\n", 1000. * (clock_end - clock_start));

  myTimer.newEvent("COPY INST TO DEV", start, main_stream);
  dev_instructions <T> ins = dev_instructions <T> (workers, &dag, &schedule, h);
  myTimer.newEvent("COPY INST TO DEV", end, main_stream);

  void ** args = ins.getLaunchArgs();

  myTimer.newEvent("GETRF", start, main_stream);
  cudaError_t error = cudaLaunchKernel((void *) kernel_dynamic <T, shm_size>, workers, num_threads, args, 0, main_stream);
  myTimer.newEvent("GETRF", end, main_stream);

  fprintf(stderr, "Kernel Launch: %s\n\n", cudaGetErrorString(error));

  h_ops dense_op = h_ops (getrf, root);
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

  cudaStreamDestroy(main_stream); */

  return cudaSuccess;
}



#endif