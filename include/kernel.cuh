
#ifndef _KERNEL_CUH
#define _KERNEL_CUH

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <intellisense.cuh>

__device__ int thread_rank()
{ return (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x; }

__device__ int block_dim()
{ return blockDim.z * blockDim.y * blockDim.x; }

__device__ int block_rank()
{ return (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x; }

__device__ int grid_dim()
{ return gridDim.z * gridDim.y * gridDim.x; }

__device__ int warp_rank()
{ return thread_rank() / warpSize; }

__device__ int lane_rank()
{ return thread_rank() - warpSize * warp_rank(); }

__device__ void wait (long long int count)
{
  long long int last = clock64();
  long long int lapse = 0;
  while (lapse < count)
  {
    long long int stamp = clock64();
    long long int interval = stamp - last;
    lapse += (interval > 0) ? interval : 0;
    last = stamp;
  }
}

__global__ void kernel_dynamic (int inst_length, const bool *dep, int *dep_counts, int *status, const long long int interval = 1000)
{
  __shared__ int inst;
  __shared__ int commit;

  if (thread_rank() == 0)
  { commit = -1; }

  while (true)
  {
    if (thread_rank() == 0)
    {
      inst = -1;
      for(int i = commit + 1; i < inst_length; i++)
      {
        if (i == commit + 1 && status[i] == -1) { commit++; }
        if (dep_counts[i] == 0 && status[i] == 0)
        {
          int s = atomicAdd(&status[i], 1);
          if (s == 0) { inst = i; break; }
        }
      }
    }
	__syncthreads();

    if (commit == inst_length - 1) 
    { break; } 
  
    if (inst >= 0) 
    {
      long long int start = clock64();
      wait(100000000);
	  __syncthreads();
      long long int end = clock64();
  
      if (thread_rank() == 0)
      {
        printf("block: %d, inst: %d, start: %lli, end: %lli\n", block_rank(), inst, start, end);
        status[inst] = -1;
        for (int i = inst + 1; i < inst_length; i++)
        { if (dep[i * inst_length + inst]) atomicSub(&dep_counts[i], 1); } 
      }
    }
    else if (thread_rank() == 0)
    {
      wait(interval);
    }
  }

}



#endif