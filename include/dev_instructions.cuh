#ifndef _DEV_INSTRUCTIONS_CUH
#define _DEV_INSTRUCTIONS_CUH

#include <pspl.cuh>

template <class T> class dev_instructions
{
private:

  int ** insts;
  T ** ptrs;
  int ** pivot_ptrs;
  int * comm_space;
  
  int workers;
  int *inst_ptr;
  int *inst_lengths;
  int ptrs_size;
  int pivot_ptrs_size;
  int comm_space_size;

public:

  __host__ dev_instructions (const int num_workers, const int default_inst_length = 1024, const int default_ptrs_size = 1024, const int default_comm_space_size = 1024)
  {
    cudaMallocManaged(&insts, num_workers * sizeof(int *), cudaMemAttachGlobal);
    cudaMallocManaged(&ptrs, default_ptrs_size * sizeof(T *), cudaMemAttachGlobal);
    cudaMallocManaged(&pivot_ptrs, default_ptrs_size * sizeof(int *), cudaMemAttachGlobal);
    cudaMallocManaged(&comm_space, default_comm_space_size * sizeof(int), cudaMemAttachGlobal);

    for (int i = 0; i < num_workers; i++)
    { 
      cudaMallocManaged(&insts[i], default_inst_length * sizeof(int), cudaMemAttachGlobal); 
      cudaMemset(insts[i], -1, default_inst_length * sizeof(int));
    }

    cudaMemset(ptrs, 0, ptrs_size * sizeof(T *));
    cudaMemset(pivot_ptrs, 0, pivot_ptrs_size * sizeof(int *));
    cudaMemset(comm_space, 0, pivot_ptrs_size * sizeof(int));

    workers = num_workers;
    inst_ptr = new int[num_workers];
    inst_lengths = new int[num_workers];
    ptrs_size = default_ptrs_size;
    pivot_ptrs_size = default_ptrs_size;
    comm_space_size = default_comm_space_size;

    memset(inst_ptr, 0, num_workers * sizeof(int));
    memset(inst_lengths, default_inst_length, num_workers * sizeof(int));

  }

  __host__ ~dev_instructions ()
  {
    for (int i = 0; i < workers; i++)
    { cudaFree(insts[i]); }
    cudaFree(insts);
    cudaFree(ptrs);
    cudaFree(pivot_ptrs);
    cudaFree(comm_space);
    delete[] inst_ptr;
    delete[] inst_lengths;
  }

  __host__ void change_ptrs_size (const int size_in)
  {
    const int size = (size_in >= 1) ? size_in : 1;
    T ** ptrs_new;
    cudaMallocManaged(&ptrs_new, size * sizeof(T *), cudaMemAttachGlobal);

    for (int i = 0; i < size; i++)
    { ptrs_new[i] = (i < ptrs_size) ? ptrs[i] : nullptr; }

    cudaFree(ptrs);
    ptrs_size = size;
    ptrs = ptrs_new;

  }

  __host__ void change_pivot_ptrs_size (const int size_in)
  {
    const int size = (size_in >= 1) ? size_in : 1;
    int ** pivot_ptrs_new;
    cudaMallocManaged(&pivot_ptrs_new, size * sizeof(T *), cudaMemAttachGlobal);

    for (int i = 0; i < size; i++)
    { pivot_ptrs_new[i] = (i < pivot_ptrs_size) ? pivot_ptrs[i] : nullptr; }

    cudaFree(pivot_ptrs);
    pivot_ptrs_size = size;
    pivot_ptrs = pivot_ptrs_new;

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

  __host__ int load_pivot_ptr (int * ptr)
  {
    int i = 0;
    while (i < pivot_ptrs_size)
    {
      if (pivot_ptrs[i] == ptr)
      { return i; }
      else if (pivot_ptrs[i] == nullptr)
      { pivot_ptrs[i] = ptr; return i; }
      else
      { i++; }
    }
    change_pivot_ptrs_size (pivot_ptrs_size * 2);
    pivot_ptrs[i] = ptr; return i;
  }

  __host__ void set_getrf_inst (const int i, T * M, const int nx, const int ny, const int ld, int * p = nullptr)
  {
    if (i >= 0 && i < inst_length)
    {
      if (insts[i] != nullptr) { cudaFree(insts[i]); }

      int *inst;
      cudaMallocManaged(&inst, 6 * sizeof(int), cudaMemAttachGlobal);
      inst[0] = (int) getrf;
      inst[1] = load_ptr(M);
      inst[2] = (p == nullptr) ? -1 : load_pivot_ptr(p);
      inst[3] = nx;
      inst[4] = ny;
      inst[5] = ld;

      insts[i] = inst;
    }
  }

  __host__ void set_trsml_inst (const int i, T * B, T * L, const int nx_b, const int ny_b, const int nx_l, const int ld_b, const int ld_l)
  {
    if (i >= 0 && i < inst_length)
    {
      if (insts[i] != nullptr) { cudaFree(insts[i]); }

      int *inst;
      cudaMallocManaged(&inst, 8 * sizeof(int), cudaMemAttachGlobal);
      inst[0] = (int) trsml;
      inst[1] = load_ptr(B);
      inst[2] = load_ptr(L);
      inst[3] = nx_b;
      inst[4] = ny_b;
      inst[5] = nx_l;
      inst[6] = ld_b;
      inst[7] = ld_l;

      insts[i] = inst;
    }
  }

  __host__ void set_trsmr_inst (const int i, T * B, T * U, const int nx_b, const int ny_b, const int ny_u, const int ld_b, const int ld_u)
  {
    if (i >= 0 && i < inst_length)
    {
      if (insts[i] != nullptr) { cudaFree(insts[i]); }

      int *inst;
      cudaMallocManaged(&inst, 8 * sizeof(int), cudaMemAttachGlobal);
      inst[0] = (int) trsmr;
      inst[1] = load_ptr(B);
      inst[2] = load_ptr(U);
      inst[3] = nx_b;
      inst[4] = ny_b;
      inst[5] = ny_u;
      inst[6] = ld_b;
      inst[7] = ld_u;

      insts[i] = inst;
    }
  }

  __host__ void set_gemm_inst (const int i, T * M, T * A, T * B, const int m, const int n, const int k, const int ld_m, const int ld_a, const int ld_b)
  {
    if (i >= 0 && i < inst_length)
    {
      if (insts[i] != nullptr) { cudaFree(insts[i]); }

      int *inst;
      cudaMallocManaged(&inst, 10 * sizeof(int), cudaMemAttachGlobal);
      inst[0] = (int) gemm;
      inst[1] = load_ptr(M);
      inst[2] = load_ptr(A);
      inst[3] = load_ptr(B);
      inst[4] = m;
      inst[5] = n;
      inst[6] = k;
      inst[7] = ld_m;
      inst[8] = ld_a;
      inst[9] = ld_b;

      insts[i] = inst;
    }
  }

  __host__ void set_pivot_inst (const int i, T * M, int * p, const int nx, const int ny, const int ld, const bool recover)
  {
    if (i >= 0 && i < inst_length)
    {
      if (insts[i] != nullptr) { cudaFree(insts[i]); }

      int *inst;
      cudaMallocManaged(&inst, 7 * sizeof(int), cudaMemAttachGlobal);
      inst[0] = (int) pivot;
      inst[1] = load_ptr(M);
      inst[2] = (p == nullptr) ? -1 : load_pivot_ptr(p);
      inst[3] = nx;
      inst[4] = ny;
      inst[5] = ld;
      inst[6] = (int) (recover);

      insts[i] = inst;
    }
  }

  __host__ void print() const
  {
    for (int i = 0; i < workers; i++)
    {
      printf("worker %d: ", i);
      for (int j = 0; j < inst_lengths[i] && insts[i][j] != -1; j++)
      { printf("%d ", insts[i][j]); }
      printf("fin.\n");
    }

  }

  /*__device__ int inst_fetch (const int look_ahead_offset, int * inst_shm)
  {

    if (thread_rank() == 0)
    {
      int i = * inst_shm;
      *inst_shm = -1;

      while (i < inst_length && * inst_shm == -1)
      {
        if (dep_counts[i] == 0 && status[i] == 0 && atomicAdd(&status[i], 1) == 0)
        { * inst_shm = i; }
        else
        { i++; }
      }
    }

    __syncthreads();
    return * inst_shm;
  }

  __device__ void inst_execute (const int inst_num, T * shm, const int shm_size)
  {
    if (inst_num >= 0)
    {
      const int * inst = insts[inst_num];
      switch ((operation_t)inst[0])
      {
      case nop:
      { break; }

      case getrf:
      {
        blockDenseGetrf_shm <T>((T *)ptrs[inst[1]], inst[3], inst[4], inst[5], (int *)ptrs[inst[2]], shm);
        break;
      }

      case trsml:
      {
        blockDenseTrsmL_shm <T>((T *)ptrs[inst[1]], (T *)ptrs[inst[2]], inst[3], inst[4], inst[5], inst[6], inst[7], shm);
        //blockDenseTrsmL_lr_shm <T> (m1, m2, inst[3], inst[4], inst[5], inst[6], inst[7], false, shm);
        break;
      }

      case trsmr:
      {
        blockDenseTrsmR_shm <T>((T *)ptrs[inst[1]], (T *)ptrs[inst[2]], inst[3], inst[4], inst[5], inst[6], inst[7], shm);
        //blockDenseTrsmR_lr_shm <T> (m1, m2, inst[3], inst[4], inst[5], inst[6], inst[7], false, shm);
        break;
      }

      case gemm:
      {
        blockDenseGemm_Cshm_RM_Sub <T> ((T *)ptrs[inst[1]], (T *)ptrs[inst[2]], (T *)ptrs[inst[3]], inst[4], inst[5], inst[6], inst[7], inst[8], inst[9], false, false, shm, shm_size);
        break;
      }

      case pivot:
      {
        blockApplyPivot <T> ((T *)ptrs[inst[1]], (int *)ptrs[inst[2]], inst[3], inst[4], inst[5], (bool)inst[6], shm, shm_size);
        break;
      }

      }

    }
    __syncthreads();

  }

  __device__ bool inst_commit (const int inst_num, int * inst_shm)
  {

    if (thread_rank() == 0)
    {
      if (inst_num >= 0)
      {
        for (int i = 1; i <= dep[inst_num][0]; i++)
        {
          atomicSub (&dep_counts[dep[inst_num][i]], 1);
        }
        status[inst_num] = -1;
      }

      * inst_shm = 0;
      while (status[* inst_shm] == -1)
      { (* inst_shm)++; }
    }

    __syncthreads();

    return (* inst_shm) < inst_length;
  }*/

  __device__ void run ()
  {
    /*__shared__ T shm[6144];
    bool looping = true;

    while (looping)
    {
      int i = inst_fetch (0, (int *) &shm[0]);
      inst_execute(i, &shm[0], 6144);
      looping = inst_commit(i, (int *) &shm[0]);
      __syncthreads();
    }*/
  }

};

#endif