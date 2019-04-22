
#ifndef _KERNEL_CUH
#define _KERNEL_CUH

#include <pspl.cuh>

template <class T, int shm_size> __global__ void __launch_bounds__ (1024, 2)
  kernel_dynamic (int ** insts, T ** ptrs, int ** pivot_ptrs, int * comm_space)
{
  __shared__ int shm [shm_size]; int * pc = insts [block_rank()], next_pc = 0;
  
load_inst:
  if (thread_rank() < _MAX_INST_LENGTH)
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
    int * p = (shm[3] == -1) ? nullptr : (int *) pivot_ptrs[shm[3]], nx = shm[4], ny = shm[5], ld = shm[6];
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
    T * M = ptrs[shm[2]];
    int * p = pivot_ptrs[shm[3]], nx = shm[4], ny = shm[5], ld = shm[6];
    bool p_T = (bool) shm[7];
    __syncthreads();
    blockApplyPivot <T> (M, p, nx, ny, ld, p_T, (T *) shm, shm_size * 4 / sizeof(T));
    next_pc = 8; goto sync;
  }

  case trsml_lr:
  {
    T * B = ptrs[shm[2]], * L = ptrs[shm[3]];
    int nx_b = shm[4], ny_b = shm[5], nx_l = shm[6], ld_b = shm[7], ld_l = shm[8];
    bool b_T = (bool) shm[9];
    __syncthreads();
    blockDenseTrsmL_lr_shm <T> (B, L, nx_b, ny_b, nx_l, ld_b, ld_l, b_T, (T *) shm);
    next_pc = 10; goto sync;
  }

  case trsmr_lr:
  {
    T * B = ptrs[shm[2]], * U = ptrs[shm[3]];
    int nx_b = shm[4], ny_b = shm[5], ny_u = shm[6], ld_b = shm[7], ld_u = shm[8];
    bool b_T = (bool) shm[9];
    __syncthreads();
    blockDenseTrsmR_lr_shm <T> (B, U, nx_b, ny_b, ny_u, ld_b, ld_u, b_T, (T *) shm);
    next_pc = 10; goto sync;  
  }

  case gemm3:
  {
    T * M = ptrs[shm[2]], * A = ptrs[shm[3]], * B = ptrs[shm[4]], * C = ptrs[shm[5]];
    int m = shm[6], n = shm[7], k = shm[8], l = shm[9], ld_m = shm[10], ld_a = shm[11], ld_b = shm[12], ld_c = shm[13];
    bool a_T = (bool) shm[14], b_T = (bool) shm[15], c_T = (bool) shm[16];
    __syncthreads();
    blockDenseGemm_3x_Cshm_RM_Sub <T> (M, A, B, C, m, n, k, l, ld_m, ld_a, ld_b, ld_c, a_T, b_T, c_T, (T *) shm, shm_size * 4 / sizeof(T));
    next_pc = 17; goto sync;
  }

  case gemm4:
  {
    T * M = ptrs[shm[2]], * A = ptrs[shm[3]], * B = ptrs[shm[4]], * C = ptrs[shm[5]], * D = ptrs[shm[6]];
    int m = shm[7], n = shm[8], k = shm[9], l = shm[10], o = shm[11], ld_m = shm[12], ld_a = shm[13], ld_b = shm[14], ld_c = shm[15], ld_d = shm[16];
    bool a_T = (bool) shm[17], b_T = (bool) shm[18], c_T = (bool) shm[19], d_T = (bool) shm[20];
    __syncthreads();
    blockDenseGemm_4x_Cshm_RM_Sub <T> (M, A, B, C, D, m, n, k, l, o, ld_m, ld_a, ld_b, ld_c, ld_d, a_T, b_T, c_T, d_T, (T *) shm, shm_size * 4 / sizeof(T));
    next_pc = 21; goto sync;
  }

  case gemm5:
  {
    T * M = ptrs[shm[2]], * A = ptrs[shm[3]], * B = ptrs[shm[4]], * C = ptrs[shm[5]], * D = ptrs[shm[6]], * E = ptrs[shm[7]];
    int m = shm[8], n = shm[9], k = shm[10], l = shm[11], o = shm[12], p = shm[13];
    int ld_m = shm[14], ld_a = shm[15], ld_b = shm[16], ld_c = shm[17], ld_d = shm[18], ld_e = shm[19];
    bool a_T = (bool) shm[20], b_T = (bool) shm[21], c_T = (bool) shm[22], d_T = (bool) shm[23], e_T = (bool) shm[24];
    __syncthreads();
    blockDenseGemm_5x_Cshm_RM_Sub <T> (M, A, B, C, D, E, m, n, k, l, o, p, ld_m, ld_a, ld_b, ld_c, ld_d, ld_e, a_T, b_T, c_T, d_T, e_T, (T *) shm, shm_size * 4 / sizeof(T));
    next_pc = 25; goto sync;
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
  const int ny = h -> getNy(), nx = h -> getNx();
  printf("Testing Hierarchical-LU for: %d x %d.\n", ny, nx);

  const h_index * root = h -> getRootIndex();
  const h_ops_tree * tree = h -> generateOps_GETRF(root);

  cudaDeviceProp deviceprop;
  cudaGetDeviceProperties(&deviceprop, 0);
  int numSMs = deviceprop.multiProcessorCount, numBlocksPerSm = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, (void *) kernel_dynamic <T, shm_size>, num_threads, 0);
  printf("# SMs: %d, # Blocks per SM for launch: %d\n", numSMs, numBlocksPerSm);

  const int workers_max = numSMs * numBlocksPerSm, workers = workers_max < num_blocks ? workers_max : num_blocks;
  if (workers == 0)
  { printf("Too many resources requested for launch.\n"); delete tree; return cudaErrorInvalidConfiguration; }
  else if (workers < num_blocks)
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

  h_ops dense_op = h_ops (getrf, root, nx, ny, 0);
  delete root;

  const unsigned long long int exeFLOPS = dag.getFops(), estFLOPS = dense_op.getFops();
  const double exeTime = myTimer.dumpAllEvents_Sync(), compressRatio = 100. * exeFLOPS / estFLOPS;

  printf("-----------------------------------------------------\n");
  printf("Actual FLOPS: %llu.\nDense-LU FLOPS: % llu.\nFLOPS Compression Ratio: %f%%.\n", exeFLOPS, estFLOPS, compressRatio);

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
  printf("FLOPS/S.\n");


  printf("-----------------------------------------------------\n");

  cudaStreamDestroy(main_stream);

  delete args;
  return error;
}



#endif