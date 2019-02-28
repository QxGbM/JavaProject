
#ifndef _KERNEL_CUH
#define _KERNEL_CUH

#include <dag.cuh>
#include <cuda.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

__device__ void wait (clock_t count)
{
  clock_t start_clock = clock();
  clock_t clock_offset = 0;
  while (clock_offset < count)
  {
    clock_offset = clock() - start_clock;
  }
}

__global__ void kernel_dynamic (int inst_length, const int *dep, int *progress, int *status, const clock_t interval = 100)
{
  const thread_block g = this_thread_block();
  
  __shared__ int inst;
  __shared__ int commit;

  if (g.thread_rank() == 0)
  { commit = -1; }

  while (true)
  {
    if (g.thread_rank() == 0)
    {
      inst = -1;
      for(int i = commit + 1; i < inst_length; i++)
      {
        if (i == commit + 1 && status[i] == -1) { commit++; }
        if (progress[i] == 0 && status[i] == 0)
        {
          int s = atomicAdd(&status[i], 1);
          if (s == 0) { inst = i; break; }
        }
      }
    }
    g.sync();

    if (commit == inst_length - 1) 
    { break; } 
  
    if (inst >= 0) 
    {
      long long int start = clock64();
      unsigned long long int a = 0;
      for (unsigned long long int i = 0; i < 10000000; i++)
      { a += i; }
      wait(100000000);
      long long int end = clock64();
  
      if (g.thread_rank() == 0)
      {
        printf("block: %d, inst: %d, start: %lli, end: %lli\n", blockIdx.x, inst, start, end);
        status[inst] = -1;
        for (int i = inst + 1; i < inst_length; i++)
        { if (dep[i * inst_length + inst] == 1) atomicSub(&progress[i], 1); } 
      }
    }
    else
    {
      wait(interval); 
    }
  }

}



#endif