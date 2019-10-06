
#pragma once
#ifndef _KERNEL_CUH
#define _KERNEL_CUH

#include <timer.cuh>
#include <dev_dense_funcs.cuh>
#include <dev_low_rank_funcs.cuh>
#include <pspl.cuh>

__global__ void kernel_dynamic (const int ** __restrict__ insts, void ** __restrict__ ptrs, volatile int * __restrict__ comm_space, 
  real_t ** __restrict__ block_tmps, real_t * __restrict__ dev_rnd_seed, unsigned long long ** __restrict__ clocks)
{
  __shared__ int shm [_SHM_SIZE]; 

  const int * pc = insts [block_rank()], t_id = thread_rank(); 
  real_t * my_tmp = block_tmps[block_rank()];
  unsigned long long * clocks_block = clocks[block_rank()];

load_inst:
  int next_pc = 0;
  const int * signal_id = nullptr;
  if (t_id < _MAX_INST_LENGTH)
  { shm[t_id] = pc[t_id]; }
  if (t_id == 0)
  { clocks_block[0] = clock64(); clocks_block = &clocks_block[1]; }
  __syncthreads();

  switch ((opcode_t) shm[0])
  {
  case execute: 
  { signal_id = &pc[1]; goto exe; }
  case signal_wait: 
  { goto wait; }
  case finish: default: 
  { goto fin; }
  }

exe:
  switch ((operation_t) shm[2])
  {
  case nop:
  { next_pc = nop_l; goto write; }
  case getrf:
  {
    real_t * M = (real_t *) ptrs[shm[3]]; 
    const int offset = shm[4], nx = shm[5], ny = shm[6], ld = shm[7];
    __syncthreads();
    blockDenseGetrf (&M[offset], nx, ny, ld, (real_t *) shm);
    next_pc = getrf_l; goto write;  
  }

  case trsml:
  {
    real_t * B = (real_t *) ptrs[shm[3]], * L = (real_t *) ptrs[shm[4]];
    const int offset_b = shm[5], offset_l = shm[6], nx_b = shm[7], ny_b = shm[8], nx_l = shm[9], ld_b = shm[10], ld_l = shm[11];
    const bool b_T = (bool) shm[12];
    __syncthreads();
    if (b_T)
    { }
    else
    { blockDenseTrsmL (&B[offset_b], &L[offset_l], nx_b, ny_b, nx_l, ld_b, ld_l, (real_t *) shm); }
    next_pc = trsml_l; goto write;
  }

  case trsmr:
  {
    real_t * B = (real_t *) ptrs[shm[3]], * U = (real_t *) ptrs[shm[4]];
    const int offset_b = shm[5], offset_u = shm[6], nx_b = shm[7], ny_b = shm[8], ny_u = shm[9], ld_b = shm[10], ld_u = shm[11];
    const bool b_T = (bool) shm[12];
    __syncthreads();
    if (b_T)
    { blockDenseTrsmR_transposeB (&B[offset_b], &U[offset_u], nx_b, ny_b, ny_u, ld_b, ld_u, (real_t *) shm); }
    else
    { blockDenseTrsmR (&B[offset_b], &U[offset_u], nx_b, ny_b, ny_u, ld_b, ld_u, (real_t *) shm); }
    next_pc = trsmr_l; goto write;
  }

  case gemm:
  {
    real_t * M = (real_t *) ptrs[shm[3]], * A = (real_t *) ptrs[shm[4]], * B = (real_t *) ptrs[shm[5]];
    const int offset_m = shm[6], offset_a = shm[7], offset_b = shm[8], m = shm[9], n = shm[10], k = shm[11], ld_m = shm[12], ld_a = shm[13], ld_b = shm[14];
    const bool a_T = (bool) shm[15], b_T = (bool) shm[16];
    __syncthreads();
    blockDenseGemm (-1., 1., &M[offset_m], &A[offset_a], &B[offset_b], m, n, k, ld_m, ld_a, ld_b, a_T, b_T, (real_t *) shm);
    next_pc = gemm_l; goto write;
  }

  case gemm_plus:
  {
    real_t * M = (real_t *) ptrs[shm[3]], * A = (real_t *) ptrs[shm[4]], * B = (real_t *) ptrs[shm[5]];
    const int offset_m = shm[6], offset_a = shm[7], offset_b = shm[8], m = shm[9], n = shm[10], k = shm[11], ld_m = shm[12], ld_a = shm[13], ld_b = shm[14];
    const bool a_T = (bool) shm[15], b_T = (bool) shm[16];
    __syncthreads();
    blockDenseGemm (1., 1., &M[offset_m], &A[offset_a], &B[offset_b], m, n, k, ld_m, ld_a, ld_b, a_T, b_T, (real_t *) shm);
    next_pc = gemm_plus_l; goto write;
  }

  case gemm_3x:
  {
    real_t * M = (real_t *) ptrs[shm[3]], * A = (real_t *) ptrs[shm[4]], * B = (real_t *) ptrs[shm[5]], * C = (real_t *) ptrs[shm[6]];
    const int offset_m = shm[7], offset_a = shm[8], offset_b = shm[9], offset_c = shm[10], m = shm[11], n = shm[12], k = shm[13], l = shm[14];
    const int ld_m = shm[15], ld_a = shm[16], ld_b = shm[17], ld_c = shm[18];
    const bool a_T = (bool) shm[19], b_T = (bool) shm[20], c_T = (bool) shm[21];
    const int control = shm[22];
    __syncthreads();
    blockDenseGemm_3x (-1., 1., &M[offset_m], &A[offset_a], &B[offset_b], &C[offset_c], m, n, k, l, ld_m, ld_a, ld_b, ld_c, a_T, b_T, c_T, control, (real_t *) shm, my_tmp);
    next_pc = gemm_3x_l; goto write;
  }

  case gemm_4x:
  {
    real_t * M = (real_t *) ptrs[shm[3]], * A = (real_t *) ptrs[shm[4]], * B = (real_t *) ptrs[shm[5]], * C = (real_t *) ptrs[shm[6]], * D = (real_t *) ptrs[shm[7]];
    const int offset_m = shm[8], offset_a = shm[9], offset_b = shm[10], offset_c = shm[11], offset_d = shm[12];
    const int m = shm[13], n = shm[14], k = shm[15], l = shm[16], o = shm[17];
    const int ld_m = shm[18], ld_a = shm[19], ld_b = shm[20], ld_c = shm[21], ld_d = shm[22];
    const bool a_T = (bool) shm[23], b_T = (bool) shm[24], c_T = (bool) shm[25], d_T = (bool) shm[26];
    const int control = shm[27], offset = shm[28];
    __syncthreads();
    blockDenseGemm_4x (-1., 1., &M[offset_m], &A[offset_a], &B[offset_b], &C[offset_c], &D[offset_d], m, n, k, l, o, ld_m, ld_a, ld_b, ld_c, ld_d, a_T, b_T, c_T, d_T, control, offset, (real_t *) shm, my_tmp);

    next_pc = gemm_4x_l; goto write;
  }

  case accum:
  {
    real_t * U1 = (real_t *) ptrs[shm[3]], * VT1 = (real_t *) ptrs[shm[4]], * U2 = (real_t *) ptrs[shm[5]], * VT2 = (real_t *) ptrs[shm[6]];
    const int offset_u1 = shm[7], offset_vt1 = shm[8], offset_u2 = shm[9], offset_vt2 = shm[10];
    const int nx = shm[11], ny = shm[12], rank1 = shm[13], rank2 = shm[14], ld_u1 = shm[15], ld_vt1 = shm[16], ld_u2 = shm[17], ld_vt2 = shm[18];
    const int offset1 = shm[19], offset2 = shm[20];
    __syncthreads();
    blockLowRankAccum (&U1[offset_u1], &VT1[offset_vt1], &U2[offset_u2], &VT2[offset_vt2], nx, ny, rank1, rank2, ld_u1, ld_vt1, ld_u2, ld_vt2, offset1, offset2, (real_t *) shm, my_tmp, dev_rnd_seed);
    next_pc = accum_l; goto write;
  }

  default: goto fin;
  }

wait:
  if (t_id == 0)
  { wait(shm[2]); shm[0] = comm_space[shm[1]]; }
  __syncthreads();
  if (shm[0])
  { next_pc = 3; }
  goto sync;

write:
  if (t_id == 0)
  { comm_space[* signal_id] = 1; }
  __threadfence();
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

void print_dev_mat (real_t * dev_mat, const int nx, const int ny)
{
   real_t * data = new real_t [nx * ny];
   cudaMemcpy (data, dev_mat, nx * ny * sizeof(real_t), cudaMemcpyDeviceToHost);
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
    cudaMalloc(&tmp[i], (1 + lengths[i]) * sizeof(unsigned long long));
    cudaMemset(tmp[i], 0, (1 + lengths[i]) * sizeof(unsigned long long));
  }
  cudaMemcpy(* clocks, tmp, workers * sizeof(unsigned long long *), cudaMemcpyHostToDevice);

  return cudaGetLastError();
}

cudaError_t generateLaunchArgsFromTree (int *** dev_insts, void *** dev_ptrs, int ** comm_space, real_t *** block_tmps, real_t ** dev_rnd_seed, unsigned long long *** clocks,
  instructions_scheduler ** schedule_addr, double * total_lapse, long long * flops, const h_ops_tree * tree, real_t ** tmp_ptrs, const int workers, const int start_index = 0, const int length_max = 0)
{
  double clock_start, clock_end, clock_lapse, clock_total = 0.;
  printf("-- Host Summary: -- \n");

  clock_start = omp_get_wtime();
  h_ops_dag dag = h_ops_dag (tree, start_index, length_max);
  clock_end = omp_get_wtime();
  clock_lapse = clock_end - clock_start;
  clock_total += clock_lapse;
  printf("DAG Created in %f ms.\n", 1000. * clock_lapse); //dag.print();

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

  clock_start = omp_get_wtime();
  cudaError_t error = ins.getLaunchArgs(dev_insts, dev_ptrs, comm_space, block_tmps, dev_rnd_seed, _SEED);
  clock_end = omp_get_wtime();
  clock_lapse = clock_end - clock_start;
  clock_total += clock_lapse;
  printf("Args generated in %f ms.\n", 1000. * clock_lapse);
  fprintf(stderr, "-- Host Args Generation: %s. --\n\n", cudaGetErrorString(error));

  * total_lapse = clock_total;
  * flops = dag.getFlops();
  return error;
}

cudaError_t launchKernelWithArgs (int ** dev_insts, void ** dev_ptrs, int * comm_space, real_t ** block_tmps, real_t * dev_rnd_seed, unsigned long long ** clocks, 
  const int workers, const int num_threads, cudaStream_t main_stream = 0)
{
  void ** args = new void * [6] { &dev_insts, &dev_ptrs, &comm_space, &block_tmps, &dev_rnd_seed, &clocks };
  cudaError_t error = cudaLaunchKernel((void *) kernel_dynamic, workers, num_threads, args, 0, main_stream);
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

cudaError_t hierarchical_GETRF (dev_hierarchical * h, const int num_blocks, const int num_threads, const int kernel_size = 0)
{
  cudaSetDevice(0);
  if (sizeof(real_t) == 8 && cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte) == cudaSuccess)
  { printf("Shared memory bank size configured to be 8-bytes.\n"); }

  cudaDeviceProp deviceprop;
  cudaGetDeviceProperties(&deviceprop, 0);
  int numSMs = deviceprop.multiProcessorCount, numBlocksPerSm = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSm, (void *) kernel_dynamic, num_threads, 0);
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
    error = generateLaunchArgsFromTree (&dev_insts, &dev_ptrs, &comm_space, &block_tmps, &dev_rnd_seed, &clocks, &schedule, &clock_lapse, &tmp, tree, tmp_ptrs, workers, i * kernel_size, kernel_size);
    printf("Host %f ms.\n\n", 1000. * clock_lapse);
    exeFLOPS += tmp;

    sprintf(event_name, "Kernel %d", i);

    myTimer.newEvent(event_name, start, main_stream);
    error = launchKernelWithArgs (dev_insts, dev_ptrs, comm_space, block_tmps, dev_rnd_seed, clocks, workers, num_threads, main_stream);
    myTimer.newEvent(event_name, end, main_stream);

    schedule -> analyzeClocks(clocks);
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



#endif