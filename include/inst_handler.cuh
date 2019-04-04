#ifndef _INST_HANDLER_CUH
#define _INST_HANDLER_CUH

#include <pspl.cuh>

template <class T> class inst_handler
{
private:
  unsigned long long int fops;

  int inst_length;
  int ** insts;
  int ** dep;
  int * dep_counts;
  int * status;
  int inst_ready;

  int ptrs_size;
  T ** ptrs;

  int pivot_ptrs_size;
  int ** pivot_ptrs;

public:

  __host__ inst_handler (const h_ops_dag * dag, const dev_hierarchical <T> *h, const int ptrs_size_in = 16, const int pivot_ptrs_size_in = 16)
  {
    fops = dag -> getFops();
    inst_length = dag -> getLength();

    cudaMallocManaged(&insts, inst_length * sizeof(int *), cudaMemAttachGlobal);
    cudaMallocManaged(&dep, inst_length * sizeof(int *), cudaMemAttachGlobal);
    cudaMallocManaged(&dep_counts, inst_length * sizeof(int), cudaMemAttachGlobal);
    cudaMallocManaged(&status, inst_length * sizeof(int), cudaMemAttachGlobal);
    inst_ready = 0;

    ptrs_size = ptrs_size_in;
    cudaMallocManaged(&ptrs, ptrs_size_in * sizeof(T *), cudaMemAttachGlobal);

    pivot_ptrs_size = pivot_ptrs_size_in;
    cudaMallocManaged(&pivot_ptrs, pivot_ptrs_size_in * sizeof(int *), cudaMemAttachGlobal);

    reset(false);

    for (int i = 0; i < inst_length; i++)
    {
      h_ops * op = dag -> getOps(i);
      switch (op -> opType())
      {
      case nop: 
        break;

      case getrf: 
        set_getrf_inst(i, op -> wr0_ptr <T> (h), op -> wr0_nx(), op -> wr0_ny(), op -> wr0_ld(), nullptr); // TODO
        break;

      case trsml:
        set_trsml_inst(i, op -> wr0_ptr <T> (h), op -> r0_ptr <T> (h), op -> wr0_nx(), op -> wr0_ny(), op -> r0_nx(), op -> wr0_ld(), op -> r0_ld());
        break;

      case trsmr:
        set_trsmr_inst(i, op -> wr0_ptr <T> (h), op -> r0_ptr <T> (h), op -> wr0_nx(), op -> wr0_ny(), op -> r0_ny(), op -> wr0_ld(), op -> r0_ld());
        break;

      case gemm:
        set_gemm_inst(i, op -> wr0_ptr <T> (h), op -> r0_ptr <T> (h), op -> r1_ptr <T> (h), op -> wr0_ny(), op->wr0_nx(), op -> r0_nx(), op -> wr0_ld(), op -> r0_ld(), op -> r1_ld());
        break;
        
      case pivot: // TODO
        break;
      }

      const int dep_length = dag -> getDepLength(i);
      cudaMallocManaged(&dep[i], (dep_length + 1) * sizeof(int), cudaMemAttachGlobal);
      dag -> flattenDep(i, dep[i]);
      dep_counts[i] = dag -> getDepCount(i);
    }
  }

  __host__ ~inst_handler ()
  {
    for (int i = 0; i < inst_length; i++)
    {
      if (insts[i] != nullptr) 
      { cudaFree(insts[i]); }
      if (dep[i] != nullptr)
      { cudaFree(dep[i]); }
    }
    cudaFree(insts);
    cudaFree(dep);
    cudaFree(dep_counts);
    cudaFree(status);
    cudaFree(ptrs);
    cudaFree(pivot_ptrs);
  }

  __host__ void reset (const bool free_insts = true)
  {
    for (int i = 0; free_insts && i < inst_length; i++)
    {
      if (insts[i] != nullptr) 
      { cudaFree(insts[i]); }
      if (dep[i] != nullptr)
      { cudaFree(dep[i]); }
    }
    cudaMemset(insts, 0, inst_length * sizeof(int *));
    cudaMemset(dep, 0, inst_length * sizeof(int *));
    cudaMemset(dep_counts, 0, inst_length * sizeof(int));
    cudaMemset(status, 0, inst_length * sizeof(int));
    cudaMemset(ptrs, 0, ptrs_size * sizeof(T *));
    cudaMemset(pivot_ptrs, 0, pivot_ptrs_size * sizeof(int *));
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
    for (int i = 0; i < inst_length; i++)
    {
      if (insts[i] != nullptr)
      {
        operation_t type = (operation_t) insts[i][0];
        printf("%d: ", i);
        switch (type)
        {
        case nop: 
          printf("NOP."); 
          break;

        case getrf: printf("GETRF M %d: %d x %d, (ld %d), P %d.", 
          insts[i][1], insts[i][4], insts[i][3], insts[i][5], insts[i][2]); 
          break;

        case trsml: printf("TRSML B %d: %d x %d, L %d: %d x %d, (ld %d %d).", 
          insts[i][1], insts[i][4], insts[i][3], insts[i][2], insts[i][4], insts[i][5], insts[i][6], insts[i][7]);
          break;

        case trsmr: printf("TRSMR B %d: %d x %d, U %d: %d x %d, (ld %d %d).", 
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

    printf("Total Float Ops: %llu\n\n", fops);

    for (int i = 0; i < inst_length; i++)
    {
      if (dep[i][0] > 0)
      {
        for (int j = 1; j <= dep[i][0]; j++)
        { printf("(%d -> %d) ", i, dep[i][j]); }
        printf("\n\n");
      }
      printf("Inst %d: [%d Output] [%d Input] dependencies.\n\n", i, dep[i][0], dep_counts[i]);
    }
  }



  __device__ int inst_fetch_right_looking (const int look_ahead_offset = 0)
  {
    __shared__ int inst_shm;

    if (thread_rank() == 0)
    {
      inst_shm = -1;
      int i = inst_ready + look_ahead_offset;

      while (i < inst_length && inst_shm == -1)
      {
        if (dep_counts[i] == 0 && status[i] == 0)
        {
          if (atomicAdd(&status[i], 1) == 0) 
          { inst_shm = i; }
        }
        i++;
      }

      i = inst_ready;
      int n = i + look_ahead_offset; n = (n > inst_length) ? inst_length : n;
      while (i < n && inst_shm == -1)
      {
        if (dep_counts[i] == 0 && status[i] == 0)
        {
          if (atomicAdd(&status[i], 1) == 0) 
          { inst_shm = i; }
        }
        i++;
      }
    }
    __syncthreads();
    return inst_shm;
  }

  __device__ void inst_execute (const int inst_num)
  {
    if (inst_num >= 0)
    {
      const int *inst = insts[inst_num];
      const operation_t op = (operation_t) inst[0];
      T *m1 = ptrs[inst[1]], *m2 = ptrs[inst[2]], *m3 = ptrs[inst[3]];
      int *p = (inst[2] == -1) ? nullptr: pivot_ptrs[inst[2]];
      switch (op)
      {
      case nop: break;

      case getrf: 
        blockDenseGetrf_shm <T, 512> (m1, inst[3], inst[4], inst[5], p); 
        break;
        
      case trsml:
        blockDenseTrsmL <T> (m1, m2, inst[3], inst[4], inst[5], inst[6], inst[7]);
        break;

      case trsmr:
        blockDenseTrsmR_shm <T, 512> (m1, m2, inst[3], inst[4], inst[5], inst[6], inst[7]);
        break;

      case gemm:
        blockDenseGemm_Cshm_RM <T, 4096> (m1, m2, m3, inst[4], inst[5], inst[6], inst[7], inst[8], inst[9]);
        break;

      case pivot:
        blockApplyPivot <T, 512> (m1, p, inst[3], inst[4], inst[5], (bool) inst[6]);
        break;

      }
      __syncthreads();

    }
  }

  __device__ void inst_commit (const int inst_num)
  {
    if (thread_rank() == 0 && inst_num >= 0)
    {
      const int length = dep[inst_num][0];
      for (int i = 1; i <= length; i++)
      {
        const int inst_i = dep[inst_num][i];
        atomicSub(&dep_counts[inst_i], 1);
      }
      status[inst_num] = -1;
    }
  }

  __device__ void inst_adjust_window()
  {
    __shared__ int inst_ready_shm;

    if (thread_rank() == 0)
    {
      int i = inst_ready;
      while (i < inst_length && status[i] == -1)
      { i++; }
      inst_ready_shm = i;
    }
    __syncthreads();

    inst_ready = inst_ready_shm;
  }

  __device__ void run (const int look_ahead_offset)
  {
    while (inst_ready < inst_length)
    {
      int i;
      if (block_rank() % 2 == 0)
      { i = inst_fetch_right_looking(look_ahead_offset); }
      else
      { i = inst_fetch_right_looking(0); }
      inst_execute(i);
      inst_commit(i);
      inst_adjust_window();
    }
  }

};

#endif