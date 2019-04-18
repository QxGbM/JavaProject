
#ifndef _KERNEL_CUH
#define _KERNEL_CUH

#include <pspl.cuh>

template <class T, int shm_size> __global__ void kernel_dynamic (int ** insts, T ** ptrs, int ** pivot_ptrs, int * comm_space)
{
  __shared__ int shm [shm_size]; int * pc = insts[block_rank()], next_pc = 0;
  
load_inst:
  if (thread_rank() < 32)
  { shm[thread_rank()] = pc[thread_rank()]; }
  next_pc = 0;
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
  case getrf:
  {
    T * M = ptrs[shm[2]]; 
    int * p = (shm[3] == -1) ? nullptr : (int *)pivot_ptrs[shm[3]];
    int nx = shm[4], ny = shm[5], ld = shm[6];
    __syncthreads();
    blockDenseGetrf_shm <T> (M, p, nx, ny, ld, (T *) shm);
    next_pc = 7; goto sync;  
  }

  case trsml:
  {
    T * B = ptrs[shm[2]], * L = ptrs[shm[3]];
    int nx_b = shm[4], ny_b = shm[5], nx_l = shm[6], ld_b = shm[7], ld_l = shm[8];
    __syncthreads();
    blockDenseTrsmL_shm <T> (B, L, nx_b, ny_b, nx_l, ld_b, ld_l, (T *) shm);
    next_pc = 9; goto sync;  
  }

  case trsmr:
  {
    T * B = ptrs[shm[2]], * U = ptrs[shm[3]];
    int nx_b = shm[4], ny_b = shm[5], ny_u = shm[6], ld_b = shm[7], ld_u = shm[8];
    __syncthreads();
    blockDenseTrsmR_shm <T> (B, U, nx_b, ny_b, ny_u, ld_b, ld_u, (T *) shm);
    next_pc = 9; goto sync;  
  }

  case gemm:
  {
    T * M = ptrs[shm[2]], * A = ptrs[shm[3]], * B = ptrs[shm[4]];
    int m = shm[5], n = shm[6], k = shm[7], ld_m = shm[8], ld_a = shm[9], ld_b = shm[10];
    bool a_T = (bool) shm[11], b_T = (bool) shm[12];
    __syncthreads();
    blockDenseGemm_Cshm_RM_Sub <T> (M, A, B, m, n, k, ld_m, ld_a, ld_b, a_T, b_T, (T *) shm, shm_size * 4 / sizeof(T));
    next_pc = 13; goto sync;
  }

  case pivot:
  {
    //blockApplyPivot <T> ((T *) ptrs[shm[2]], (int *) pivot_ptrs[shm[3]], shm[4], shm[5], shm[6], (bool) shm[7], (T *) shm, 49152 / sizeof(T));
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
  const h_ops_tree * tree = h -> generateOps_GETRF();

  cudaDeviceProp deviceprop;
  cudaGetDeviceProperties(&deviceprop, 0);
  int numSMs = deviceprop.multiProcessorCount, numBlocksPerSm = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, (void *) kernel_dynamic <T, shm_size>, num_threads, 0);
  printf("# SMs: %d, # Blocks per SM for launch: %d\n", numSMs, numBlocksPerSm);

  const int workers_max = numSMs * numBlocksPerSm, workers = workers_max < num_blocks ? workers_max : num_blocks;
  if (workers_max < num_blocks)
  { printf("Number of launched blocks reduced from %d to %d. \n", num_blocks, workers); }

  h_ops_dag dag = h_ops_dag (tree);
  delete tree;

  inst_scheduler schedule = inst_scheduler (&dag, workers);

  dev_instructions <T> ins = dev_instructions <T> (workers, &dag, &schedule, h);

  cudaStream_t main_stream;
  cudaStreamCreate(&main_stream);

  if (sizeof(T) == 8) 
  {
    printf("Shared memory bank size configured to be 8-bytes.\n"); 
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  }

  void ** args = ins.getLaunchArgs();

  timer myTimer = timer();

  myTimer.newEvent("GETRF", start, main_stream);
  cudaError_t error = cudaLaunchKernel((void *) kernel_dynamic <T, shm_size>, workers, num_threads, args, 0, main_stream);
  myTimer.newEvent("GETRF", end, main_stream);

  fprintf(stderr, "Kernel Launch: %s\n\n", cudaGetErrorString(error));
  error = myTimer.dumpAllEvents_Sync(dag.getFops());

  cudaStreamDestroy(main_stream);

  delete args;
  return error;
}



#endif