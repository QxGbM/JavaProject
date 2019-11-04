
#pragma once
#ifndef _INSTRUCTIONS_MANAGER_CUH
#define _INSTRUCTIONS_MANAGER_CUH

#include <definitions.cuh>

class instructions_manager
{
private:

  int ** insts;
  int workers;
  int * inst_lengths;

  void ** ptrs;
  int ptrs_size;

  int * tmp_sizes;
  int rnd_size;

  int comm_length;

  void changeInstsSize (const int worker_id, const int size_in)
  {
    if (worker_id >= 0 && worker_id < workers)
    {
      const int size = (size_in >= 1) ? size_in : 1, size_old = inst_lengths[worker_id];
      if (size != size_old)
      {
        int * inst_new = new int [size], * inst_old = insts[worker_id], size_ = (size > size_old) ? size_old : size;
        memset(inst_new, -1, size * sizeof(int));

        for (int i = 0; i < size_; i++)
        { inst_new[i] = inst_old[i]; }

        delete[] insts[worker_id];
        insts[worker_id] = inst_new;
        inst_lengths[worker_id] = size;
      }
    }
  }

  void changePointersSize (const int size_in)
  {
    const int size = (size_in >= 1) ? size_in : 1;
    if (size != ptrs_size)
    {
      void ** ptrs_new = new void * [size];
      int size_ = (size > ptrs_size) ? ptrs_size : size;
      memset(ptrs_new, 0, size * sizeof(void *));

      for (int i = 0; i < size_; i++)
      { ptrs_new[i] = ptrs[i]; }

      delete[] ptrs;
      ptrs_size = size;
      ptrs = ptrs_new;
    }
  }

  void loadPointers (void ** ptrs_in, const int n_ptrs, int * mapping)
  {
    if (n_ptrs == 0)
    { return; }

    int count = 0, last = ptrs_size;
    for (int i = 0; i < ptrs_size; i++)
    {
      void * ptr = ptrs[i];
      if (ptr == nullptr)
      { last = i; break; }

      for (int j = 0; j < n_ptrs; j++)
      {
        if (ptr == ptrs_in[j]) 
        { mapping[j] = i; count ++; }
      }
    }

    if (count < n_ptrs)
    { 
      while (last + n_ptrs - count >= ptrs_size)
      { changePointersSize (ptrs_size * 2); }
    
      count = 0;
      for (int i = 0; i < n_ptrs; i++)
      {
        if (mapping[i] == -1 && ptrs_in[i] != nullptr)
        {
          for (int j = 0; j < i; j++) 
          {
            if (ptrs_in[j] == ptrs_in[i])
            { mapping[i] = mapping[j]; }
          }

          if (mapping[i] == -1)
          { ptrs[last + count] = ptrs_in[i]; mapping[i] = last + count; count ++; }
        } 
      }
    }

  }

  int loadInsts (int * tmp_size, int * rnd_size, const int worker_id, const instructions_queue * queue, const h_ops_dag * dag, void ** tmp_ptrs, const double gpu_clock_multiplier = _CLOCK_MULTIPLIER)
  {
    if (queue == nullptr) 
    { insts[worker_id][0] = (int) finish; return 1; }

    int loc = 0, size = 0, rnd = 0, size_max = 0, rnd_max = 0, * inst = &(insts[worker_id][loc]), n_ptrs = 16, * mapping = new int [n_ptrs]; 
    void ** ptrs = new void * [n_ptrs];

    for (const instructions_queue * ptr = queue; ptr != nullptr; ptr = ptr -> getNext())
    {
      const int signal_id = ptr -> getInst();

      if (ptr -> getExW())
      {
        const h_ops * op = dag -> getOp (signal_id);
        while (loc + _MAX_INST_LENGTH >= inst_lengths[worker_id])
        { changeInstsSize(worker_id, inst_lengths[worker_id] * 2); inst = &(insts[worker_id][loc]); }

        inst[0] = (int) execute;
        inst[1] = signal_id;

        memset(mapping, -1, n_ptrs * sizeof(int));
        n_ptrs = op -> getDataPointers (ptrs, tmp_ptrs);

        #pragma omp critical
        { loadPointers (ptrs, n_ptrs, mapping); }

        const int t = op -> writeOpParametersTo (&inst[2], &size, &rnd, mapping);

        if (size > size_max)
        { size_max = size; }
        if (rnd > rnd_max)
        { rnd_max = rnd; }

        loc += t;
        inst = &inst[t];
      }
      else
      {
        while (loc + 3 >= inst_lengths[worker_id])
        { changeInstsSize(worker_id, inst_lengths[worker_id] * 2); inst = &(insts[worker_id][loc]); }

        double wait = gpu_clock_multiplier * ptr -> getElapsedFlops();

        inst[0] = (int) signal_wait;
        inst[1] = signal_id;
        inst[2] = 0x7fffffff & (int) wait;
      
        loc += 3;
        inst = &inst[3];
      }
    }

    if (loc + 1 >= inst_lengths[worker_id])
    { changeInstsSize(worker_id, inst_lengths[worker_id] + 1); inst = &(insts[worker_id][loc]); }

    inst[0] = (int) finish;

    delete[] mapping;
    delete[] ptrs;
    * tmp_size = size_max;

    #pragma omp critical
    { 
      if (* rnd_size < rnd_max) 
      { * rnd_size = rnd_max; } 
    }

    return loc + 1;
  }

public:

  instructions_manager (const int num_workers, const h_ops_dag * dag, const instructions_scheduler * schedule, void ** tmp_ptrs)
  {
    insts = new int * [num_workers];
    workers = num_workers;
    inst_lengths = new int [num_workers];

    ptrs = new void * [_PTRS_LENGTH];
    ptrs_size = _PTRS_LENGTH;

    tmp_sizes = new int[num_workers];
    rnd_size = 0;

#pragma omp parallel for
    for (int i = 0; i < num_workers; i++)
    { 
      insts[i] = new int [_INSTS_LENGTH];
      memset (insts[i], -1, _INSTS_LENGTH * sizeof(int));
      inst_lengths[i] = _INSTS_LENGTH;
    }

    memset(ptrs, 0, ptrs_size * sizeof(void *));
    memset(tmp_sizes, 0, num_workers * sizeof(int));

#pragma omp parallel for
    for (int i = 0; i < num_workers; i++)
    { inst_lengths[i] = loadInsts(&tmp_sizes[i], &rnd_size, i, schedule -> getSchedule(i), dag, tmp_ptrs); }

    comm_length = dag -> getLength();
  }

  ~instructions_manager ()
  {
#pragma omp parallel for
    for (int i = 0; i < workers; i++)
    { delete[] insts[i]; }

    delete[] insts;
    delete[] inst_lengths;
    delete[] ptrs;
    delete[] tmp_sizes;
  }

  cudaError_t getLaunchArgs (int *** dev_insts, void *** dev_ptrs, int ** comm_space, real_t*** block_tmps, real_t ** dev_rnd_seed, const unsigned int seed_in = 0) const
  {
    int ** insts_temp = new int * [workers]; real_t ** tmps_temp = new real_t * [workers];
    cudaMalloc(dev_insts, workers * sizeof(int *));
    cudaMalloc(block_tmps, workers * sizeof(real_t *));

    for (int i = 0; i < workers; i++)
    {
      cudaMalloc(&insts_temp[i], ((size_t) inst_lengths[i] + _MAX_INST_LENGTH) * sizeof(int));
      cudaMemcpy(insts_temp[i], insts[i], (size_t) inst_lengths[i] * sizeof(int), cudaMemcpyHostToDevice);

      if (tmp_sizes[i] > 0)
      {
        cudaMalloc(&tmps_temp[i], (size_t) tmp_sizes[i] * real_bits);
        cudaMemset(tmps_temp[i], 0, (size_t) tmp_sizes[i] * real_bits);
      }
      else
      { tmps_temp[i] = nullptr; }
    }

    cudaMemcpy(* dev_insts, insts_temp, workers * sizeof(int *), cudaMemcpyHostToDevice);
    cudaMemcpy(* block_tmps, tmps_temp, workers * sizeof(real_t *), cudaMemcpyHostToDevice);
    delete[] insts_temp; delete[] tmps_temp;

    cudaMalloc(dev_ptrs, ptrs_size * sizeof(void *));
    cudaMemcpy(* dev_ptrs, ptrs, ptrs_size * sizeof(void *), cudaMemcpyHostToDevice);

    cudaMalloc(comm_space, comm_length * sizeof(int));
    cudaMemset(* comm_space, 0, comm_length * sizeof(int));

    if (seed_in > 0)
    { srand(seed_in); }

    real_t * rnd_seed = new real_t[rnd_size];

#pragma omp parallel for
    for (int i = 0; i < rnd_size; i++) 
    { rnd_seed[i] = (real_t) rand() / RAND_MAX; }

    cudaMalloc(dev_rnd_seed, (size_t) rnd_size * real_bits);
    cudaMemcpy(* dev_rnd_seed, rnd_seed, (size_t) rnd_size * real_bits, cudaMemcpyHostToDevice);
    delete[] rnd_seed;
    
    return cudaGetLastError();
  }

  void print (const int limit = 32, const int ptrs_limit = 32) const
  {
    for (int i = 0; i < workers; i++)
    {
      printf("worker %d(%d): ", i, inst_lengths[i]);
      for (int j = 0; j < inst_lengths[i] && j < limit; j++)
      { printf("%d ", insts[i][j]); }
      if (inst_lengths[i] > limit)
      { printf("...\n"); }
      else
      { printf("fin.\n"); }
    }

    for (int i = 0; i < ptrs_size && i < ptrs_limit; i++)
    { printf("%d: %p\n", i, ptrs[i]); }

  }

};

#endif