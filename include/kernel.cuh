
#ifndef _KERNEL_CUH
#define _KERNEL_CUH

#include <pspl.cuh>

template <class T> __global__ void kernel_dynamic (int ** insts, T ** ptrs, int ** pivot_ptrs, int * comm_space)
{
  __shared__ int shm [12288]; int * pc = insts[block_rank()], next_pc = 0;
  
load_inst:
  if (thread_rank() < 32)
  { shm[thread_rank()] = pc[thread_rank()]; }
  next_pc = 0;
  __syncthreads();

  if (thread_rank() == 0) 
  { printf("%d %d %d\n", block_rank(), shm[0], shm[1]); }
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
  __syncthreads();
  switch ((operation_t) shm[1])
  {
  case getrf:
  {
    blockDenseGetrf_shm <T> ((T *) ptrs[shm[2]], (shm[3] == -1) ? nullptr : (int *) pivot_ptrs[shm[3]], shm[4], shm[5], shm[6], (T *) shm);
    next_pc = 7; goto sync;  
  }

  case trsml:
  {
    blockDenseTrsmL_shm <T> ((T *) ptrs[shm[2]], (T *) ptrs[shm[3]], shm[4], shm[5], shm[6], shm[7], shm[8], (T *) shm);
    next_pc = 9; goto sync;  
  }

  case trsmr:
  {
    blockDenseTrsmR_shm <T> ((T *) ptrs[shm[2]], (T *) ptrs[shm[3]], shm[4], shm[5], shm[6], shm[7], shm[8], (T *) shm);
    next_pc = 9; goto sync;  
  }

  case gemm:
  {
    blockDenseGemm_Cshm_RM_Sub <T> ((T *) ptrs[shm[2]], (T *) ptrs[shm[3]], (T *) ptrs[shm[4]], shm[5], shm[6], shm[7], 
      shm[8], shm[9], shm[10], (bool) shm[11], (bool) shm[12], (T *) shm, 49152 / sizeof(T));
    next_pc = 13; goto sync;
  }

  case pivot:
  {
    blockApplyPivot <T> ((T *) ptrs[shm[2]], (int *) pivot_ptrs[shm[3]], shm[4], shm[5], shm[6], (bool) shm[7], (T *) shm, 49152 / sizeof(T));
    next_pc = 8; goto sync;
  }

  default: goto fin;
  }

wait:
  __syncthreads();
  if (comm_space[shm[1]])
  { next_pc = 2; }
  goto sync;

write:
  __syncthreads();
  comm_space[shm[1]] = 1;
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

template <class T> __host__ cudaError_t hierarchical_GETRF (dev_hierarchical <T> * h, const int num_blocks, const int num_threads)
{
  const h_ops_tree * tree = h -> generateOps_GETRF();

  h_ops_dag dag = h_ops_dag (tree);
  delete tree;

  inst_scheduler schedule = inst_scheduler (&dag, num_blocks);
  schedule.print();

  dev_instructions <T> ins = dev_instructions <T> (num_blocks, &dag, &schedule, h);

  cudaStream_t main_stream;
  cudaStreamCreate(&main_stream);

  if (sizeof(T) == 8) 
  {
    printf("shared memory bank size set for double precision.\n"); 
    cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
  }

  void ** args = ins.getLaunchArgs();

  timer myTimer = timer();

  myTimer.newEvent("GETRF", start, main_stream);
  cudaError_t error = cudaLaunchKernel((void *) kernel_dynamic <T>, num_blocks, num_threads, args, 0, main_stream);
  myTimer.newEvent("GETRF", end, main_stream);

  fprintf(stderr, "Kernel Launch: %s\n\n", cudaGetErrorString(error));
  error = myTimer.dumpAllEvents_Sync(dag.getFops());

  cudaStreamDestroy(main_stream);

  return error;
}



#endif