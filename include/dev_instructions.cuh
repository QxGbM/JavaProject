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
  int comm_space_used;
  int comm_space_size;

  __host__ cudaError_t changeInstsSize (const int worker_id, const int size_in)
  {
    if (worker_id < 0 || worker_id >= workers) 
    { return cudaErrorUnsupportedLimit; }

    const int size = (size_in >= 1) ? size_in : 1;
    if (size != inst_lengths[worker_id])
    {
      int * inst_new;
      cudaMallocManaged(&inst_new, size * sizeof(int), cudaMemAttachGlobal);
      cudaMemset(inst_new, -1, size * sizeof(int));
      cudaMemcpy(inst_new, insts[worker_id], (size > inst_lengths[worker_id] ? inst_lengths[worker_id] : size) * sizeof(int), cudaMemcpyDeviceToDevice);

      cudaFree(insts[worker_id]);
      inst_lengths[worker_id] = size;
      insts[worker_id] = inst_new;
      return cudaGetLastError();
    }
    return cudaSuccess;
  }

  __host__ cudaError_t changePointersSize (const int size_in)
  {
    const int size = (size_in >= 1) ? size_in : 1;
    if (size != ptrs_size)
    {
      T ** ptrs_new;
      cudaMallocManaged(&ptrs_new, size * sizeof(T *), cudaMemAttachGlobal);
      cudaMemset(ptrs_new, 0, size * sizeof(T *));
      cudaMemcpy(ptrs_new, ptrs, (size > ptrs_size ? ptrs_size : size) * sizeof(T *), cudaMemcpyDeviceToDevice);

      cudaFree(ptrs);
      ptrs_size = size;
      ptrs = ptrs_new;
      return cudaGetLastError();
    }
    return cudaSuccess;
  }

  
  __host__ cudaError_t changePivotPointersSize (const int size_in)
  {
    const int size = (size_in >= 1) ? size_in : 1;
    if (size != pivot_ptrs_size)
    {
      int ** ptrs_new;
      cudaMallocManaged(&ptrs_new, size * sizeof(int *), cudaMemAttachGlobal);
      cudaMemset(ptrs_new, 0, size * sizeof(int *));
      cudaMemcpy(ptrs_new, pivot_ptrs, (size > pivot_ptrs_size ? pivot_ptrs_size : size) * sizeof(int *), cudaMemcpyDeviceToDevice);

      cudaFree(pivot_ptrs);
      pivot_ptrs_size = size;
      pivot_ptrs = ptrs_new;
      return cudaGetLastError();
    }
    return cudaSuccess;
  }

  __host__ cudaError_t changeCommSpaceSize (const int size_in)
  {
    const int size = (size_in >= 1) ? size_in : 1;
    if (size != pivot_ptrs_size)
    {
      int * comm_space_new;
      cudaMallocManaged(&comm_space_new, size * sizeof(int), cudaMemAttachGlobal);
      cudaMemset(comm_space_new, 0, size * sizeof(int));
      cudaMemcpy(comm_space_new, comm_space, (size > comm_space_size ? comm_space_size : size) * sizeof(int), cudaMemcpyDeviceToDevice);

      cudaFree(comm_space);
      comm_space_size = size;
      comm_space = comm_space_new;
      return cudaGetLastError();
    }
    return cudaSuccess;
  }

  __host__ int loadPointer (T * ptr)
  {
    if (ptr == nullptr) { return -1; }

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
    changePointersSize (ptrs_size * 2);
    ptrs[i] = ptr; return i;
  }

  __host__ int loadPivotPointer (int * ptr)
  {
    if (ptr == nullptr) { return -1; }

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
    changePivotPointersSize (pivot_ptrs_size * 2);
    pivot_ptrs[i] = ptr; return i;
  }

  __host__ void execGetrf (const int worker_id, T * M, const int nx, const int ny, const int ld, int * p = nullptr)
  {
    if (worker_id >= 0 && worker_id < workers)
    {
      const int inst_length = 7, loc = inst_ptr[worker_id];
      while (loc + inst_length >= inst_lengths[worker_id] - 1)
      { changeInstsSize(worker_id, inst_lengths[worker_id] * 2); }
      int * inst = &(insts[worker_id][loc]);

      inst[0] = (int) execute;
      inst[1] = (int) getrf;
      inst[2] = loadPointer(M);
      inst[3] = loadPivotPointer(p);
      inst[4] = nx;
      inst[5] = ny;
      inst[6] = ld;
      inst[7] = (int) finish;

      inst_ptr[worker_id] = loc + inst_length;
    }
  }

  __host__ void execTrsmL (const int worker_id, T * B, T * L, const int nx_b, const int ny_b, const int nx_l, const int ld_b, const int ld_l)
  {
    if (worker_id >= 0 && worker_id < workers)
    {
      const int inst_length = 9, loc = inst_ptr[worker_id];
      while (loc + inst_length >= inst_lengths[worker_id] - 1)
      { changeInstsSize(worker_id, inst_lengths[worker_id] * 2); }
      int * inst = &(insts[worker_id][loc]);

      inst[0] = (int) execute;
      inst[1] = (int) trsml;
      inst[2] = loadPointer(B);
      inst[3] = loadPivotPointer(L);
      inst[4] = nx_b;
      inst[5] = ny_b;
      inst[6] = nx_l;
      inst[7] = ld_b;
      inst[8] = ld_l;
      inst[9] = (int) finish;

      inst_ptr[worker_id] = loc + inst_length;
    }
  }

  __host__ void execTrsmR (const int worker_id, T * B, T * U, const int nx_b, const int ny_b, const int ny_u, const int ld_b, const int ld_u)
  {
    if (worker_id >= 0 && worker_id < workers)
    {
      const int inst_length = 9, loc = inst_ptr[worker_id];
      while (loc + inst_length >= inst_lengths[worker_id] - 1)
      { changeInstsSize(worker_id, inst_lengths[worker_id] * 2); }
      int * inst = &(insts[worker_id][loc]);

      inst[0] = (int) execute;
      inst[1] = (int) trsmr;
      inst[2] = loadPointer(B);
      inst[3] = loadPointer(U);
      inst[4] = nx_b;
      inst[5] = ny_b;
      inst[6] = ny_u;
      inst[7] = ld_b;
      inst[8] = ld_u;
      inst[9] = (int) finish;

      inst_ptr[worker_id] = loc + inst_length;
    }
  }

  __host__ void execGemm (const int worker_id, T * M, T * A, T * B, const int m, const int n, const int k, const int ld_m, const int ld_a, const int ld_b)
  {
    if (worker_id >= 0 && worker_id < workers)
    {
      const int inst_length = 11, loc = inst_ptr[worker_id];
      while (loc + inst_length >= inst_lengths[worker_id] - 1)
      { changeInstsSize(worker_id, inst_lengths[worker_id] * 2); }
      int * inst = &(insts[worker_id][loc]);

      inst[0] = (int) execute;
      inst[1] = (int) gemm;
      inst[2] = loadPointer(M);
      inst[3] = loadPointer(A);
      inst[4] = loadPointer(B);
      inst[5] = m;
      inst[6] = n;
      inst[7] = k;
      inst[8] = ld_m;
      inst[9] = ld_a;
      inst[10] = ld_b;
      inst[11] = (int) finish;

      inst_ptr[worker_id] = loc + inst_length;
    }
  }

  __host__ void execPivot (const int i, T * M, int * p, const int nx, const int ny, const int ld, const bool recover)
  {
    if (i >= 0 && i < inst_length)
    {
      const int inst_length = 8, loc = inst_ptr[worker_id];
      while (loc + inst_length >= inst_lengths[worker_id] - 1)
      { changeInstsSize(worker_id, inst_lengths[worker_id] * 2); }
      int * inst = &(insts[worker_id][loc]);

      inst[0] = (int) execute;
      inst[1] = (int) pivot;
      inst[2] = loadPointer(M);
      inst[3] = loadPivotPointer(p);
      inst[4] = nx;
      inst[5] = ny;
      inst[6] = ld;
      inst[7] = (int) recover;
      inst[8] = (int) finish;

      inst_ptr[worker_id] = loc + inst_length;
    }
  }

public:

  __host__ dev_instructions (const int num_workers, const int default_inst_length = 1024, const int default_ptrs_size = 1024, const int default_comm_space_size = 1024)
  {
    cudaMallocManaged(&insts, num_workers * sizeof(int *), cudaMemAttachGlobal);
    cudaMallocManaged(&ptrs, default_ptrs_size * sizeof(T *), cudaMemAttachGlobal);
    cudaMallocManaged(&pivot_ptrs, default_ptrs_size * sizeof(int *), cudaMemAttachGlobal);
    cudaMallocManaged(&comm_space, default_comm_space_size * sizeof(int), cudaMemAttachGlobal);

    workers = num_workers;
    inst_ptr = new int[num_workers];
    inst_lengths = new int[num_workers];
    ptrs_size = default_ptrs_size;
    pivot_ptrs_size = default_ptrs_size;
    comm_space_size = default_comm_space_size;
    comm_space_used = 0;

    for (int i = 0; i < num_workers; i++)
    { 
      cudaMallocManaged(&insts[i], default_inst_length * sizeof(int), cudaMemAttachGlobal); 
      cudaMemset(insts[i], -1, default_inst_length * sizeof(int));
      inst_lengths[i] = default_inst_length;
    }

    cudaMemset(ptrs, 0, ptrs_size * sizeof(T *));
    cudaMemset(pivot_ptrs, 0, pivot_ptrs_size * sizeof(int *));
    cudaMemset(comm_space, 0, pivot_ptrs_size * sizeof(int));
    memset(inst_ptr, 0, num_workers * sizeof(int));

    print();
    execGetrf(0, nullptr, 4, 4, 4);

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


  __host__ void print() const
  {
    for (int i = 0; i < workers; i++)
    {
      printf("worker %d(%d): ", i, inst_lengths[i]);
      for (int j = 0; j < inst_ptr[i] + 1; j++)
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