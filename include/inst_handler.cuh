#ifndef _INST_HANDLER_CUH
#define _INST_HANDLER_CUH

#include <definitions.cuh>
#include <kernel.cuh>

template <class T> class inst_handler
{
private:
  int inst_length;
  int ** insts;
  bool * dep;

  int ptrs_size;
  T ** ptrs;

  int commit;
  int * dep_counts;
  int * status;

public:
  __host__ inst_handler (const int inst_length_in, const int ptrs_size_in = 16)
  {
    inst_length = inst_length_in;

    cudaMallocManaged(&insts, inst_length_in * sizeof(int *), cudaMemAttachGlobal);
    cudaMallocManaged(&dep, inst_length_in * inst_length_in * sizeof(bool), cudaMemAttachGlobal);

    ptrs_size = ptrs_size_in;
    cudaMallocManaged(&ptrs, ptrs_size_in * sizeof(T *), cudaMemAttachGlobal);

    commit = -1;
    cudaMallocManaged(&dep_counts, inst_length_in * sizeof(int), cudaMemAttachGlobal);
    cudaMallocManaged(&status, inst_length_in * sizeof(int), cudaMemAttachGlobal);

  }

  __host__ ~inst_handler()
  {
    cudaFree(insts);
    cudaFree(dep);
    cudaFree(ptrs);
    cudaFree(dep_counts);
    cudaFree(status);
  }

  __host__ void change_ptrs_size (const int size_in)
  {
    const int size = (size_in >= 1) ? size_in : 1;
    T ** ptrs_new;
    cudaMallocManaged(&ptrs_new, size * sizeof(T *), cudaMemAttachGlobal);

    for (int i = 0; i < size && i < ptrs_size; i++)
    { ptrs_new[i] = ptrs[i]; }

    cudaFree(ptrs);
    ptrs = ptrs_new;
    ptrs_size = size;
  }

  __device__ int inst_fetch ()
  {
    __shared__ int inst_shm;
    __shared__ int commit_shm;
    if (thread_rank() == 0)
    {
      inst_shm = -1;
      for (int i = commit + 1; i < inst_length; i++)
      {
        if (i == commit + 1 && status[i] == -1) 
        { commit++; }
        if (dep_counts[i] == 0 && status[i] == 0)
        {
          int s = atomicAdd(&status[i], 1);
          if (s == 0) { inst_shm = i; break; }
        }
      }
      commit_shm = commit;
    }
    __syncthreads();
    commit = commit_shm;
    return inst_shm;
  }

  __device__ void inst_execute (int inst)
  {
    if (inst >= 0)
    {
      printf("%d: %d", thread_rank(), inst);
    }
    else
    {

    }
  }

  __device__ void func()
  {
    int i = inst_fetch();
    printf("%d: %d, %d\n", thread_rank(), i, commit);
  }
};

#endif