#ifndef _INST_HANDLER_CUH
#define _INST_HANDLER_CUH

#include <pspl.cuh>

template <class T> class inst_handler
{
private:
  int inst_length;
  int ** insts;
  bool * dep;

  int ptrs_size;
  T ** ptrs;

  int inst_ready;
  int * dep_counts;
  int * status;

public:
  __host__ inst_handler (const int inst_length_in, const int ptrs_size_in = 16)
  {
    inst_length = inst_length_in;

    cudaMallocManaged(&insts, inst_length_in * sizeof(int *), cudaMemAttachGlobal);
    cudaMallocManaged(&dep, inst_length_in * inst_length_in * sizeof(bool), cudaMemAttachGlobal);
    cudaMemset(insts, 0, inst_length_in * sizeof(int *));
    cudaMemset(dep, 0, inst_length_in * inst_length_in * sizeof(bool));

    ptrs_size = ptrs_size_in;
    cudaMallocManaged(&ptrs, ptrs_size_in * sizeof(T *), cudaMemAttachGlobal);
    cudaMemset(ptrs, 0, ptrs_size_in * sizeof(T *));

    inst_ready = 0;
    cudaMallocManaged(&dep_counts, inst_length_in * sizeof(int), cudaMemAttachGlobal);
    cudaMallocManaged(&status, inst_length_in * sizeof(int), cudaMemAttachGlobal);
    cudaMemset(dep_counts, 0, inst_length_in * sizeof(int));
    cudaMemset(status, 0, inst_length_in * sizeof(int));

  }

  __host__ ~inst_handler ()
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

  __host__ int load_ptr (T * ptr)
  {
    int i = 0;
    while (i < ptrs_size)
    {
      if (ptrs[i] == ptr) 
      { return i; }
      else if (ptrs[i] == nullptr) 
      { ptrs[i] = ptr; return i; }
      else
      { i++; }
    }
    change_ptrs_size (ptrs_size * 2);
    ptrs[i] = ptr; return i;
  }

  __host__ void set_getrf_inst (const int i, T * dev_matrix, const int nx, const int ny, const int ld)
  {
    if (insts[i] != nullptr) { cudaFree(insts[i]); }

    int *inst;
    cudaMallocManaged(&inst, 5 * sizeof(int), cudaMemAttachGlobal);
    inst[0] = (int) getrf;
    inst[1] = load_ptr(dev_matrix);
    inst[2] = nx;
    inst[3] = ny;
    inst[4] = ld;

    insts[i] = inst;
  }

  __host__ void print() const
  {
    for (int i = 0; i < inst_length; i++)
    {
      if (insts[i] != nullptr)
      {
        matrix_op_t type = (matrix_op_t) insts[i][0];
        printf("%d: ", i);
        switch (type)
        {
        case nop: printf("NOP\n"); break;
        case getrf: printf("GETRF %d nx = %d, ny = %d, ld = %d.", insts[i][1], insts[i][2], insts[i][3], insts[i][4]); break;
        default: break;
        }
        printf("\n");
      }
    }
  }

  __device__ int inst_fetch ()
  {
    __shared__ int inst_shm;
    __shared__ int inst_ready_shm;
    if (thread_rank() == 0)
    {
      inst_shm = -1;
      for (int i = inst_ready; i < inst_length; i++)
      {
        if (i == inst_ready && status[i] == -1) 
        { inst_ready ++; }
        if (dep_counts[i] == 0 && status[i] == 0)
        {
          int s = atomicAdd(&status[i], 1);
          if (s == 0) { inst_shm = i; break; }
        }
      }
      inst_ready_shm = inst_ready;
    }
    __syncthreads();
    inst_ready = inst_ready_shm;
    return inst_shm;
  }

  __device__ void inst_execute (int inst_num)
  {
    if (inst_num >= 0)
    {
      int *inst = insts[inst_num];
      if (thread_rank() == 0) printf("%d %d %d %d\n", inst[0], inst[2], inst[3], inst[4]);
      __syncthreads();

    }
  }

  __device__ void inst_commit (int inst_num)
  {
    if (thread_rank() == 0 && inst_num >= 0)
    {
      status[inst_num] = -1;
      for (int i = inst_num + 1; i < inst_length; i++)
      {
        if (dep[i * inst_length + inst_num]) 
        { atomicSub(&dep_counts[i], 1); }
      }
    }
  }

  __device__ void inst_wait (long long int count)
  {
    if (thread_rank() == 0)
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
    __syncthreads();
  }

  __device__ void run()
  {
    while (inst_ready < inst_length)
    {
      int i = inst_fetch();
      inst_execute(i);
      inst_commit(i);
      inst_wait(1000);
    }
  }
};

#endif