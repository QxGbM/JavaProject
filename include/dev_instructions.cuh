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

  __host__ int loadOperationPointers (int * inst, const h_ops * op, const dev_hierarchical <T> * h)
  {
    int l1 = op -> l_wr(), l2 = op -> l_r(), t = 0;
    for (int i = 0; i < l1; i++)
    {
      inst[t] = loadPointer(op -> wr_ptr <T>(i, h)); t++;
      if (op -> opType() == getrf)
      { inst[t] = loadPivotPointer(op -> wr_pivot_ptr(i, h)); t++; }
    }
    for (int i = 0; i < l2; i++)
    {
      if (op -> opType() == pivot)
      { inst[t] = loadPivotPointer(op -> r_pivot_ptr(i, h)); t++; }
      else
      { inst[t] = loadPointer(op -> r_ptr <T> (i, h)); t++; }
    }
    return t;
  }

  __host__ void execOperation (const int worker_id, const h_ops * op, const dev_hierarchical <T> * h)
  {
    const int loc = inst_ptr[worker_id];
    while (loc + 16 >= inst_lengths[worker_id])
    { changeInstsSize(worker_id, inst_lengths[worker_id] * 2); }
    int * inst = &(insts[worker_id][loc]), t = 2;

    inst[0] = (int) execute;
    inst[1] = (int) op -> opType();
    
    t += loadOperationPointers (&inst[t], op, h);
    t += op -> writeParametersTo(&inst[t]);

    inst[t] = (int) finish;

    inst_ptr[worker_id] = loc + t;
  }

  __host__ void newSignalWrite (const int worker_id, const int signal_id)
  {
    if (worker_id >= 0 && worker_id < workers)
    {
      const int inst_length = 2, loc = inst_ptr[worker_id];
      while (loc + inst_length >= inst_lengths[worker_id] - 1)
      { changeInstsSize(worker_id, inst_lengths[worker_id] * 2); }
      while (signal_id >= comm_space_size)
      { changeCommSpaceSize(comm_space_size * 2); }
      int * inst = &(insts[worker_id][loc]);

      inst[0] = (int) signal_write;
      inst[1] = signal_id;
      inst[2] = (int) finish;
      
      comm_space_used ++;
      inst_ptr[worker_id] = loc + inst_length;
    }
  }

  __host__ void newSignalWait (const int worker_id, const int signal_id)
  {
    if (worker_id >= 0 && worker_id < workers)
    {
      const int inst_length = 2, loc = inst_ptr[worker_id];
      while (loc + inst_length >= inst_lengths[worker_id] - 1)
      { changeInstsSize(worker_id, inst_lengths[worker_id] * 2); }
      while (signal_id >= comm_space_size)
      { changeCommSpaceSize(comm_space_size * 2); }
      int * inst = &(insts[worker_id][loc]);

      inst[0] = (int) signal_wait;
      inst[1] = signal_id;
      inst[2] = (int) finish;
      
      inst_ptr[worker_id] = loc + inst_length;
    }
  }

  __host__ void loadInsts (const int worker_id, const inst_queue * queue, const h_ops_dag * dag, const dev_hierarchical <T> * h)
  {
    if (queue == nullptr) 
    { insts[worker_id][0] = (int) finish; return; }

    for (const inst_queue * ptr = queue; ptr != nullptr; ptr = ptr -> getNext())
    {
      int inst_n = ptr -> getInst();
      if (ptr -> getExW())
      { execOperation(worker_id, dag -> getOp(inst_n), h); newSignalWrite(worker_id, inst_n); }
      else
      { newSignalWait(worker_id, inst_n); }
    }
  }

public:

  __host__ dev_instructions (const int num_workers, const h_ops_dag * dag, const inst_scheduler * schedule, const dev_hierarchical <T> * h,
    const int default_inst_length = 1024, const int default_ptrs_size = 1024, const int default_comm_space_size = 1024)
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

    for (int i = 0; i < num_workers; i++)
    { loadInsts(i, schedule -> getSchedule(i), dag, h); }

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

  __host__ inline void ** getLaunchArgs ()
  { return new void *[4]{ &insts, &ptrs, &pivot_ptrs, &comm_space }; }


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

};

#endif