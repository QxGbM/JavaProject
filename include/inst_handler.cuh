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

    for (int i = 0; i < size; i++)
    { ptrs_new[i] = (i < ptrs_size) ? ptrs[i] : nullptr; }

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

  __host__ void add_dep (const int inst_from, const int inst_to)
  {
    if (!dep[inst_from * inst_length + inst_to]) { dep_counts[inst_to] ++; }
    dep[inst_from * inst_length + inst_to] = true;
  }

  __host__ void remove_dep(const int inst_from, const int inst_to)
  {
    if (!dep[inst_from * inst_length + inst_to]) { dep_counts[inst_to] --; }
    dep[inst_from * inst_length + inst_to] = false;
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

  __host__ void set_getrf_inst (const int i, T * M, const int nx, const int ny, const int ld)
  {
    if (insts[i] != nullptr) { cudaFree(insts[i]); }

    int *inst;
    cudaMallocManaged(&inst, 5 * sizeof(int), cudaMemAttachGlobal);
    inst[0] = (int) getrf;
    inst[1] = load_ptr(M);
    inst[2] = nx;
    inst[3] = ny;
    inst[4] = ld;

    insts[i] = inst;
  }

  __host__ void set_trsml_inst (const int i, T * L, T * B, const int nx_l, const int ny_l, const int nx_b, const int ld_l, const int ld_b)
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

  __host__ void set_trsmr_inst (const int i, T * U, T * B, const int nx_u, const int ny_u, const int ny_b, const int ld_u, const int ld_b)
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

  __host__ void set_gemm_inst (const int i, T * M, T * A, T * B, const int m, const int n, const int k, const int ld_m, const int ld_a, const int ld_b)
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

  __host__ void set_pivot_inst (const int i, T * M, int * p, const int nx, const int ny, const int ld)
  {
    // TODO
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
        case getrf: printf("GETRF M %d: %d x %d, (ld %d).", 
          insts[i][1], insts[i][3], insts[i][2], insts[i][4]); break;
        case trsml: printf("TRSML L %d: %d x %d, B %d: %d x %d, (ld %d %d).", 
          insts[i][1], insts[i][4], insts[i][3], insts[i][2], insts[i][4], insts[i][5], insts[i][6], insts[i][7]); break;
        case trsmr: printf("TRSMR U %d: %d x %d, B %d: %d x %d, (ld %d %d).", 
          insts[i][1], insts[i][4], insts[i][3], insts[i][2], insts[i][5], insts[i][3], insts[i][6], insts[i][7]); break;
        case gemm: printf("GEMM M %d: %d x %d, A %d: %d x %d, B %d: %d x %d, (ld %d %d %d)", 
          insts[i][1], insts[i][4], insts[i][5], insts[i][2], insts[i][4], insts[i][6], insts[i][3], insts[i][6], insts[i][5], insts[i][7], insts[i][8], insts[i][9]); 
          break;
        default: break;
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
      printf("Total: %d dependencies.\n", dep_counts[i]);
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
      switch (op)
      {
      case nop: break;

      case getrf: 
        blockDenseGetrf(m1, inst[2], inst[3], inst[4]); 
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
        // TODO
        break;

      default: break;
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
      if (thread_rank() == 0) { printf("%d: %d\n", block_rank(), i);  }
      inst_execute(i);
      inst_commit(i);
      inst_wait(1000);
    }
  }

};

#endif