#ifndef _INST_HANDLER_CUH
#define _INST_HANDLER_CUH

#include <pspl.cuh>

template <class T> class inst_handler
{
private:
  int inst_length;
  int ** insts;
  bool * dep;
  int * dep_counts;
  int * status;
  int inst_ready;

  int ptrs_size;
  T ** ptrs;

  int pivot_ptrs_size;
  int ** pivot_ptrs;

public:
  __host__ void reset (const bool free_insts = true)
  {
    for (int i = 0; free_insts && i < inst_length; i++)
    { if (insts[i] != nullptr) cudaFree(insts[i]); }
    cudaMemset(insts, 0, inst_length * sizeof(int *));
    cudaMemset(dep, 0, inst_length * inst_length * sizeof(bool));
    cudaMemset(dep_counts, 0, inst_length * sizeof(int));
    cudaMemset(status, 0, inst_length * sizeof(int));
    cudaMemset(ptrs, 0, ptrs_size * sizeof(T *));
    cudaMemset(pivot_ptrs, 0, pivot_ptrs_size * sizeof(int *));
  }

  __host__ inst_handler (const int inst_length_in, const int ptrs_size_in = 16, const int pivot_ptrs_size_in = 16)
  {
    inst_length = inst_length_in;

    cudaMallocManaged(&insts, inst_length_in * sizeof(int *), cudaMemAttachGlobal);
    cudaMallocManaged(&dep, inst_length_in * inst_length_in * sizeof(bool), cudaMemAttachGlobal);
    cudaMallocManaged(&dep_counts, inst_length_in * sizeof(int), cudaMemAttachGlobal);
    cudaMallocManaged(&status, inst_length_in * sizeof(int), cudaMemAttachGlobal);
    inst_ready = 0;

    ptrs_size = ptrs_size_in;
    cudaMallocManaged(&ptrs, ptrs_size_in * sizeof(T *), cudaMemAttachGlobal);

    pivot_ptrs_size = pivot_ptrs_size_in;
    cudaMallocManaged(&pivot_ptrs, pivot_ptrs_size_in * sizeof(int *), cudaMemAttachGlobal);

    reset(false);
  }

  __host__ ~inst_handler ()
  {
    for (int i = 0; i < inst_length; i++)
    { if (insts[i] != nullptr) cudaFree(insts[i]); }
    cudaFree(insts);
    cudaFree(dep);
    cudaFree(dep_counts);
    cudaFree(status);
    cudaFree(ptrs);
    cudaFree(pivot_ptrs);
  }

  __host__ void change_insts_length (const int length_in)
  {
    const int length = (length_in >= 1) ? length_in : 1;
    int ** insts_new, * dep_counts_new, * status_new;
    bool * dep_new;

    cudaMallocManaged(&insts_new, length * sizeof(int *), cudaMemAttachGlobal);
    cudaMallocManaged(&dep_new, length * length * sizeof(bool), cudaMemAttachGlobal);
    cudaMallocManaged(&dep_counts_new, length * sizeof(int), cudaMemAttachGlobal);
    cudaMallocManaged(&status_new, length * sizeof(int), cudaMemAttachGlobal);

    for (int i = 0; i < length; i++)
    {
      insts_new[i] = insts[i];
      status_new[i] = status[i];
      for (int j = 0; j < i; j++)
      {
        if (dep[j * inst_length + i])
        { dep_new[j * length + i] = true; dep_counts_new[i]++; }
      }
    }

    for (int i = length; i < inst_length; i++)
    { if (insts[i] != nullptr) cudaFree(insts[i]); }
    cudaFree(insts);
    cudaFree(dep);
    cudaFree(dep_counts);
    cudaFree(status);

    inst_length = length;
    insts = insts_new;
    dep = dep_new;
    dep_counts = dep_counts_new;
    status = status_new;

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

  __host__ void add_dep (const int inst_from, const int inst_to)
  {
    if (inst_from < inst_to)
    {
      if (!dep[inst_from * inst_length + inst_to]) { dep_counts[inst_to] ++; }
      dep[inst_from * inst_length + inst_to] = true;
    }
  }

  __host__ void remove_dep(const int inst_from, const int inst_to)
  {
    if (inst_from < inst_to)
    {
      if (!dep[inst_from * inst_length + inst_to]) { dep_counts[inst_to] --; }
      dep[inst_from * inst_length + inst_to] = false;
    }
  }

  __host__ void fill_nop_inst ()
  {
    for (int i = 0; i < inst_length; i++)
    {
      if (insts[i] == nullptr) 
      {
        cudaMallocManaged(&insts[i], sizeof(int), cudaMemAttachGlobal);
        insts[i][0] = (int) nop;
      }
    }
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

  __host__ void set_trsml_inst (const int i, T * L, T * B, const int nx_l, const int ny_l, const int nx_b, const int ld_l, const int ld_b)
  {
    if (i >= 0 && i < inst_length)
    {
      if (insts[i] != nullptr) { cudaFree(insts[i]); }

      int *inst;
      cudaMallocManaged(&inst, 8 * sizeof(int), cudaMemAttachGlobal);
      inst[0] = (int) trsml;
      inst[1] = load_ptr(L);
      inst[2] = load_ptr(B);
      inst[3] = nx_l;
      inst[4] = ny_l;
      inst[5] = nx_b;
      inst[6] = ld_l;
      inst[7] = ld_b;

      insts[i] = inst;
    }
  }

  __host__ void set_trsmr_inst (const int i, T * U, T * B, const int nx_u, const int ny_u, const int ny_b, const int ld_u, const int ld_b)
  {
    if (i >= 0 && i < inst_length)
    {
      if (insts[i] != nullptr) { cudaFree(insts[i]); }

      int *inst;
      cudaMallocManaged(&inst, 8 * sizeof(int), cudaMemAttachGlobal);
      inst[0] = (int) trsmr;
      inst[1] = load_ptr(U);
      inst[2] = load_ptr(B);
      inst[3] = nx_u;
      inst[4] = ny_u;
      inst[5] = ny_b;
      inst[6] = ld_u;
      inst[7] = ld_b;

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
    for (int i = 0; i < inst_length; i++)
    {
      if (insts[i] != nullptr)
      {
        matrix_op_t type = (matrix_op_t) insts[i][0];
        printf("%d: ", i);
        switch (type)
        {
        case nop: 
          printf("NOP."); 
          break;

        case getrf: printf("GETRF M %d: %d x %d, (ld %d), P %d.", 
          insts[i][1], insts[i][4], insts[i][3], insts[i][5], insts[i][2]); 
          break;

        case trsml: printf("TRSML L %d: %d x %d, B %d: %d x %d, (ld %d %d).", 
          insts[i][1], insts[i][4], insts[i][3], insts[i][2], insts[i][4], insts[i][5], insts[i][6], insts[i][7]); 
          break;

        case trsmr: printf("TRSMR U %d: %d x %d, B %d: %d x %d, (ld %d %d).", 
          insts[i][1], insts[i][4], insts[i][3], insts[i][2], insts[i][5], insts[i][3], insts[i][6], insts[i][7]); 
          break;

        case gemm: printf("GEMM M %d: %d x %d, A %d: %d x %d, B %d: %d x %d, (ld %d %d %d)", 
          insts[i][1], insts[i][4], insts[i][5], insts[i][2], insts[i][4], insts[i][6], insts[i][3], insts[i][6], insts[i][5], insts[i][7], insts[i][8], insts[i][9]); 
          break;

        case pivot: printf("PIVOT M %d: %d x %d, (ld %d), P %d",
          insts[i][1], insts[i][4], insts[i][3], insts[i][5], insts[i][2]);
          if (insts[i][6] == 0) printf(" APPLY.");
          else printf(" RECOVERY.");
          break;

        }
        printf("\n");
      }
    }

    for (int i = 0; i < inst_length; i++)
    {
      for (int j = 0; j < i; j++)
      {
        if (dep[j * inst_length + i])
        {
          printf("Dependency: from %d to %d.\n", j, i);
        }
      }
      printf("Inst %d Total: %d dependencies.\n", i, dep_counts[i]);
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

  __device__ void inst_execute (const int inst_num)
  {
    if (inst_num >= 0)
    {
      const int *inst = insts[inst_num];
      const matrix_op_t op = (matrix_op_t) inst[0];
      T *m1 = ptrs[inst[1]], *m2 = ptrs[inst[2]], *m3 = ptrs[inst[3]];
      int *p = (inst[2] == -1) ? nullptr: pivot_ptrs[inst[2]];
      switch (op)
      {
      case nop: break;

      case getrf: 
        blockDenseGetrf(m1, inst[3], inst[4], inst[5], p); 
        break;
        
      case trsml:
        blockDenseTrsmL(m1, m2, inst[3], inst[4], inst[5], inst[6], inst[7]);
        break;

      case trsmr:
        blockDenseTrsmR(m1, m2, inst[3], inst[4], inst[5], inst[6], inst[7]);
        break;

      case gemm:
        blockDenseGemm(m1, m2, m3, inst[4], inst[5], inst[6], inst[7], inst[8], inst[9]);
        break;

      case pivot:
        blockApplyPivot(m1, p, inst[3], inst[4], inst[5], (bool) inst[6]);
        break;

      }
      __syncthreads();

    }
  }

  __device__ void inst_commit (const int inst_num)
  {
    if (thread_rank() == 0 && inst_num >= 0)
    {
      for (int i = inst_num + 1; i < inst_length; i++)
      {
        if (dep[inst_num * inst_length + i])
        { atomicSub(&dep_counts[i], 1); }
      }
      status[inst_num] = -1;
    }
  }

  __device__ void inst_wait (const long long int count)
  {
    if (thread_rank() == 0)
    {
      long long int last = clock64();
      long long int lapse = 0;
      while (lapse < count)
      {
        const long long int stamp = clock64();
        const long long int interval = stamp - last;
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