#ifndef _INSTRUCTIONS_SCHEDULER_CUH
#define _INSTRUCTIONS_SCHEDULER_CUH

#include <pspl.cuh>

class instructions_queue
{
private:
  int inst;
  bool ex_w;
  long long int starting_flops_count;
  instructions_queue * next;

public:
  __host__ instructions_queue (const int inst_in, const bool ex_w_in, const long long int flops_in)
  {
    inst = inst_in;
    ex_w = ex_w_in;
    starting_flops_count = flops_in;
    next = nullptr;
  }

  __host__ ~instructions_queue ()
  { delete next; }

  __host__ int getInst () const
  { return inst; }

  __host__ bool getExW () const
  { return ex_w; }

  __host__ long long int getElapsedFlops () const
  { return next -> starting_flops_count - starting_flops_count; }

  __host__ instructions_queue * getNext () const
  { return next; }

  __host__ instructions_queue * setNext (const int inst_in, const bool ex_w_in, const long long int flops_in)
  {
    instructions_queue * ptr = new instructions_queue (inst_in, ex_w_in, flops_in);
    ptr -> next = next; next = ptr; return ptr;
  }

  __host__ void print() const
  {
    for (const instructions_queue * ptr = this; ptr != nullptr; ptr = ptr -> next) 
    {
      if (ptr -> ex_w) 
      { printf("i%d ", ptr -> inst); }
      else
      { printf("w%d ", ptr -> inst); }
    }
    printf("\n");
  }

};

class ready_queue 
{
private:
  int inst_number;
  int num_deps;
  long long int anticipated_flops;
  ready_queue * next;

public:
  __host__ ready_queue (const int inst_in, const int n_deps_in, const long long int flops_in, ready_queue * next_q = nullptr)
  {
    inst_number = inst_in;
    num_deps = n_deps_in;
    anticipated_flops = flops_in;
    next = next_q;
  }

  __host__ ~ready_queue ()
  { delete next; }

  __host__ int getNumDeps () const
  { return num_deps; }

  __host__ void hookup (const int inst_in, const int n_deps_in, const long long int flops_in)
  {
    ready_queue * ptr = next, * last = this;

    while (ptr != nullptr && ptr -> num_deps > n_deps_in) 
    { last = ptr; ptr = ptr -> next; }

    while (ptr != nullptr && ptr -> anticipated_flops >= flops_in && ptr -> num_deps == n_deps_in)
    { last = ptr; ptr = ptr -> next; }

    last -> next = new ready_queue (inst_in, n_deps_in, flops_in, ptr);
  }

  __host__ ready_queue * deleteFirst (int * inst_out, long long int * flops_out)
  {
    if (this == nullptr)
    { * inst_out = -1; * flops_out = 0; return nullptr; }
    else
    { 
      * inst_out = inst_number; * flops_out = anticipated_flops;
      ready_queue * ptr = next; next = nullptr;
      delete this; return ptr;
    }
  }

  __host__ void print() const
  {
    int length = 0;
    printf("Ready Queue: \n");
    for (const ready_queue * ptr = this; ptr != nullptr; ptr = ptr -> next) 
    { printf("i%d ", ptr -> inst_number); length ++; }
    printf("\nLength: %d. \n", length);
  }

};

class instructions_scheduler
{
private:
  int length;
  int workers;

  ready_queue * working_queue;
  instructions_queue ** result_queues;
  instructions_queue ** result_queue_tails;

  int * inward_deps_counter;
  long long int * flops_after_inst;
  long long int * flops_worker;

  int * inst_executed_by;
  long long int * last_sync_flops;

  __host__ void loadWorkingQueue (const h_ops_dag * dag)
  {
    for (int i = 0; i < length; i++)
    {
      if (inward_deps_counter[i] == 0)
      {
        const int num_deps = dag -> getDepCount_From(i);
        const long long int flops = dag -> getFlops(i);

        if (working_queue == nullptr)
        { working_queue = new ready_queue (i, num_deps, flops); }
        else if (working_queue -> getNumDeps() < num_deps)
        { working_queue = new ready_queue (i, num_deps, flops, working_queue); }
        else
        { working_queue -> hookup(i, num_deps, flops); }
      }
    }
  }

  __host__ long long int findLatestSyncs (const h_ops_dag * dag, int * sync_with, const int inst)
  {
    const dependency_linked_list * list = dag -> getDepList_To(inst);

    long long int * flops_container = new long long int[workers], max_sync = 0;
    
    memset(flops_container, 0, workers * sizeof(long long int));
    memset(sync_with, 0xffffffff, workers * sizeof(int));

    while (list != nullptr)
    {
      const dependency_t dep = list -> getDep();
      if (dep > no_dep)
      {
        const int inst_from = list -> getInst(), exe_worker = inst_executed_by[inst_from];
        const long long int flops = flops_after_inst[inst_from];
        if (flops_container[exe_worker] < flops)
        { flops_container[exe_worker] = flops; sync_with[exe_worker] = inst_from; }
        if (max_sync < flops)
        { max_sync = flops; }
      }
      list = list -> getNext();
    }

    delete[] flops_container;
    return max_sync;

  }

  __host__ int findWorkerWithMinimalWaiting (const long long int max_sync)
  {
    int worker_before_sync = -1, worker_after_sync = 0;
    long long int latest_before_sync = -1, earliest_after_sync = flops_worker[0];

    for (int i = 0; i < workers; i++)
    {
      const long long int flops = flops_worker[i];

      if (flops <= max_sync && flops > latest_before_sync)
      { worker_before_sync = i; latest_before_sync = flops; }
      else if (flops > max_sync && flops < earliest_after_sync)
      { worker_after_sync = i; earliest_after_sync = flops; }
    }

    if (worker_before_sync >= 0)
    { return worker_before_sync; }
    else
    { return worker_after_sync; }

  }

  __host__ void eliminateExtraSync (int * sync_with, const int worker_id)
  {
    sync_with[worker_id] = -1;

    for (int i = 0; i < workers; i++)
    {
      const int inst = sync_with[i];
      if (inst >= 0)
      {
        const long long int flops = flops_after_inst[inst];
        if (flops <= last_sync_flops[worker_id * workers + i])
        { sync_with[i] = -1; }
      }
    }

  }

  __host__ void addInstToWorker (const int inst, const long long int flops_anticipated, const int worker_id)
  {
    if (result_queues[worker_id] == nullptr)
    { result_queue_tails[worker_id] = result_queues[worker_id] = new instructions_queue(inst, true, flops_worker[worker_id]); }
    else
    { result_queue_tails[worker_id] = result_queue_tails[worker_id] -> setNext(inst, true, flops_worker[worker_id]); }

    flops_after_inst[inst] = (flops_worker[worker_id] += flops_anticipated);
    inst_executed_by[inst] = worker_id;
  }

  __host__ void addWaitToWorker (const int inst, const int worker_id)
  {
    if (result_queues[worker_id] == nullptr)
    { result_queue_tails[worker_id] = result_queues[worker_id] = new instructions_queue(inst, false, flops_worker[worker_id]); }
    else
    { result_queue_tails[worker_id] = result_queue_tails[worker_id] -> setNext(inst, false, flops_worker[worker_id]); }

    const int src = inst_executed_by[inst]; const long long int flops = flops_after_inst[inst];
    if (flops_worker[worker_id] < flops)
    { flops_worker[worker_id] = flops; }
    last_sync_flops[worker_id * workers + src] = flops;
  }

  __host__ void updateDepsCounts (const h_ops_dag * dag, const int inst_finished)
  {
    for (const dependency_linked_list * dep_list = dag -> getDepList_From(inst_finished); dep_list != nullptr; dep_list = dep_list -> getNext())
    {
      if (dep_list -> getDep() > no_dep)
      {
        const int i = dep_list -> getInst();
        if (--inward_deps_counter[i] == 0)
        {
          const int num_deps = dag -> getDepCount_From(i);
          const long long int flops = dag -> getFlops(i);

          if (working_queue == nullptr)
          { working_queue = new ready_queue (i, num_deps, flops); }
          else if (working_queue -> getNumDeps() < num_deps)
          { working_queue = new ready_queue (i, num_deps, flops, working_queue); }
          else
          { working_queue -> hookup(i, num_deps, flops); }
        }
      }
    }
  }

  __host__ void schedule (const h_ops_dag * dag)
  {
    int comm_wait_counts = 0, * sync_with = new int[workers];

    loadWorkingQueue(dag);

    for (int scheduled_insts = 0; scheduled_insts < length; scheduled_insts ++)
    {
      int inst; long long int flops;
      working_queue = working_queue -> deleteFirst(&inst, &flops);

      const long long int max_sync = findLatestSyncs(dag, sync_with, inst);
      const int worker_id = findWorkerWithMinimalWaiting(max_sync);

      eliminateExtraSync(sync_with, worker_id);

      for (int i = 0; i < workers; i++)
      {
        const int sync_i = sync_with[i];

        if (sync_i >= 0)
        { addWaitToWorker(sync_i, worker_id); comm_wait_counts ++; }
      }

      addInstToWorker(inst, flops, worker_id);

      updateDepsCounts(dag, inst);
    }

    delete[] sync_with;

    long long flops_max_worker = 0, flops_total = dag -> getFlops();

    for (int i = 0; i < workers; i++)
    {
      const long long flops_worker_i = flops_worker[i];
      if (flops_worker_i > flops_max_worker)
      { flops_max_worker = flops_worker_i; }
    }

    const double heaviest_worker_to_total = 100. * flops_max_worker / flops_total;
    const double flops_prll = 100. * flops_total / flops_max_worker / workers;
    const double comm_per_inst = 1. * comm_wait_counts / length;

    printf("-- Scheduler Summary --\n"
      "Total # of Instructions: %d. \n"
      "Percent Heaviest Loaded Worker to Total Ratio: %f%%. \n"
      "Degree of Flops Parallelism: %f%%. \n"
      "Avg. # of Communications per Instruction: %f. \n\n", 
      length, heaviest_worker_to_total, flops_prll, comm_per_inst);

  }

public:

  __host__ instructions_scheduler (const h_ops_dag * dag, const int num_workers_limit)
  {
    length = dag -> getLength();
    workers = num_workers_limit;
    working_queue = nullptr;

    result_queues = new instructions_queue * [workers];
    result_queue_tails = new instructions_queue * [workers];

    flops_after_inst = new long long int [length];
    flops_worker = new long long int [workers];

    inst_executed_by = new int [length];
    last_sync_flops = new long long int [workers * workers];

#pragma omp parallel for
    for (int i = 0; i < workers; i++)
    { result_queues[i] = nullptr; result_queue_tails[i] = nullptr; }

    inward_deps_counter = dag -> getDepCountList_To();

    memset(flops_after_inst, 0, length * sizeof(long long int));
    memset(flops_worker, 0, workers * sizeof(long long int));

    memset(inst_executed_by, 0xffffffff, length * sizeof(int));
    memset(last_sync_flops, 0, workers * workers * sizeof(long long int));

    schedule (dag);
  }

  __host__ ~instructions_scheduler ()
  {
    delete working_queue;

    for (int i = 0; i < workers; i++)
    { delete result_queues[i]; }

    delete[] result_queues;
    delete[] result_queue_tails;

    delete[] inward_deps_counter;
    delete[] flops_after_inst;
    delete[] flops_worker;

    delete[] inst_executed_by;
    delete[] last_sync_flops;
  }

  __host__ instructions_queue * getSchedule (const int worker_id) const
  { return (worker_id >= 0 && worker_id < workers) ? result_queues[worker_id] : nullptr; }

  __host__ void print () const
  {
    working_queue -> print();
    for (int i = 0; i < workers; i++)
    { printf("Worker #%d: ", i); result_queues[i] -> print(); printf("flops: %lld \n", flops_worker[i]); }
  }

};

#endif