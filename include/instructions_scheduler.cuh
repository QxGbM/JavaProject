
#pragma once
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
  instructions_queue (const int inst_in, const bool ex_w_in, const long long int flops_in)
  {
    inst = inst_in;
    ex_w = ex_w_in;
    starting_flops_count = flops_in;
    next = nullptr;
  }

  ~instructions_queue ()
  { delete next; }

  int getInst () const
  { return inst; }

  bool getExW () const
  { return ex_w; }

  long long int getElapsedFlops () const
  { return next -> starting_flops_count - starting_flops_count; }

  instructions_queue * getNext () const
  { return next; }

  instructions_queue * setNext (const int inst_in, const bool ex_w_in, const long long int flops_in)
  {
    instructions_queue * ptr = new instructions_queue (inst_in, ex_w_in, flops_in);
    ptr -> next = next; next = ptr; return ptr;
  }

  int getLength() const
  {
    int l = 0;
    for (const instructions_queue * ptr = this; ptr != nullptr; ptr = ptr -> next) 
    { l++; }
    return l;
  }

  void print() const
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

  long long int max_sync;
  int * sync_with;
  ready_queue * next;

public:
  ready_queue (const int inst_in, const int workers, const h_ops_dag * dag, const long long int * flops_after_inst, const int * inst_executed_by, ready_queue * next_q = nullptr)
  {
    inst_number = inst_in;
    num_deps = dag -> getDepCount_From(inst_in);
    anticipated_flops = dag -> getFlops_Trim(inst_in);
    next = next_q;

    const dependency_linked_list * list = dag -> getDepList_To(inst_in);

    long long int * flops_container = new long long int[workers];
    sync_with = new int[workers];
    max_sync = 0;
    
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
  }

  ~ready_queue ()
  {
    delete[] sync_with;
    delete next; 
  }

  int getInst() const
  { return inst_number; }

  int getNumDeps () const
  { return num_deps; }

  long long int getFlops (const long long int min_flops = _MIN_INST_FLOPS) const
  { return anticipated_flops > min_flops ? anticipated_flops : min_flops; }

  long long int getMaxSync() const
  { return max_sync; }

  int * getSyncWith() const
  { return sync_with; }

  ready_queue * getLast()
  {
    for (ready_queue * ptr = this; ptr != nullptr; ptr = ptr -> next)
    { 
      if (ptr -> next == nullptr) 
      { return ptr; } 
    }
    return nullptr;
  }

  ready_queue * setNext (ready_queue * next_q)
  {
    ready_queue * ptr = next;
    next = next_q;
    return ptr;
  }

  bool hookup (ready_queue * queue)
  {
    ready_queue * q_ptr = queue, * ptr = next, * last = this;

    while (q_ptr != nullptr)
    {
      long long int q_start = q_ptr -> max_sync;

      while (ptr != nullptr && ptr -> max_sync <= q_start) 
      { last = ptr; ptr = ptr -> next; }

      ready_queue * t_ptr = q_ptr;
      q_ptr = q_ptr -> next;
      t_ptr -> next = ptr;
      last -> next = t_ptr;
      last = t_ptr;
    }

    return ptr == nullptr;
  }

  ready_queue * deleteCriticalNode (ready_queue ** deleted_ptr_out, const long long int flops_synced)
  {
    if (flops_synced < max_sync)
    {
      ready_queue * ptr = next;
      next = nullptr;
      * deleted_ptr_out = this;
      return ptr;
    }
    else
    {
      int max_n = num_deps;
      ready_queue * ptr = next, * last = this, * target = this, * target_prev = nullptr;

      while (ptr != nullptr && ptr -> max_sync <= flops_synced) 
      {
        if (ptr -> num_deps > max_n)
        { target_prev = last; target = ptr; max_n = ptr -> num_deps; }
        last = ptr; ptr = ptr -> next;
      }

      if (target_prev == nullptr)
      {
        ptr = next;
        next = nullptr;
        * deleted_ptr_out = this;
        return ptr;
      }
      else
      {
        target_prev -> next = target -> next;
        target -> next = nullptr;
        * deleted_ptr_out = target;
        return this;
      }

    }
  }

  void print() const
  {
    int length = 0;
    printf("Ready Queue: \n");
    for (const ready_queue * ptr = this; ptr != nullptr; ptr = ptr -> next) 
    { printf("i%d: flops %lld start %lld. \n", ptr -> inst_number, ptr -> anticipated_flops, ptr -> max_sync); length ++; }
    printf("\nLength: %d. \n", length);
  }

};

class instructions_scheduler
{
private:
  int length;
  int workers;

  ready_queue * working_queue;
  ready_queue * working_queue_tail;

  instructions_queue ** result_queues;
  instructions_queue ** result_queue_tails;

  int * inward_deps_counter;
  long long int * flops_after_inst;
  long long int * flops_worker;

  int * inst_executed_by;
  long long int * last_sync_flops;

  long long int getSmallestLoad()
  { 
    long long int min_flops = flops_worker[0];
    for (int i = 1; i < workers; i++)
    {
      if (flops_worker[i] < min_flops)
      { min_flops = flops_worker[i]; }
    }
    return min_flops;
  }

  void loadWorkingQueue (const h_ops_dag * dag)
  {
    for (int i = 0; i < length; i++)
    {
      if (inward_deps_counter[i] == 0)
      {
        working_queue = new ready_queue(i, workers, dag, flops_after_inst, inst_executed_by, working_queue);
        if (working_queue_tail == nullptr)
        { working_queue_tail = working_queue; }
      }
    }

  }

  long long int findLatestSyncs (const h_ops_dag * dag, int * sync_with, const int inst)
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

  int findWorkerWithMinimalWaiting (const long long int max_sync)
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

  void eliminateExtraSync (int * sync_with, const int worker_id)
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

  void addInstToWorker (const int inst, const long long int flops_anticipated, const int worker_id)
  {
    if (result_queues[worker_id] == nullptr)
    { result_queue_tails[worker_id] = result_queues[worker_id] = new instructions_queue(inst, true, flops_worker[worker_id]); }
    else
    { result_queue_tails[worker_id] = result_queue_tails[worker_id] -> setNext(inst, true, flops_worker[worker_id]); }

    flops_after_inst[inst] = (flops_worker[worker_id] += flops_anticipated);
    inst_executed_by[inst] = worker_id;
  }

  void addWaitToWorker (const int inst, const int worker_id)
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

  void updateDepsCounts (const h_ops_dag * dag, const int inst_finished)
  {
    ready_queue * queue = nullptr, * last = nullptr;

    for (const dependency_linked_list * dep_list = dag -> getDepList_From(inst_finished); dep_list != nullptr; dep_list = dep_list -> getNext())
    {
      if (dep_list -> getDep() > no_dep)
      {
        const int i = dep_list -> getInst();
        if (--inward_deps_counter[i] == 0)
        {
          ready_queue * entry = new ready_queue (i, workers, dag, flops_after_inst, inst_executed_by);

          if (queue == nullptr)
          { queue = entry; last = queue; }
          else if (entry -> getMaxSync() <= queue -> getMaxSync())
          { entry -> setNext(queue); queue = entry; }
          else
          { 
            bool update_last = queue -> hookup(entry);
            if (update_last)
            { last = entry; }
          }

        }
      }
    }

    if (working_queue == nullptr)
    { 
      working_queue = queue; 
      working_queue_tail = last;
    }
    else if (queue != nullptr)
    { 
      bool update_last = working_queue -> hookup(queue);
      if (update_last)
      { working_queue_tail = last; }
    }

  }

  void schedule (const h_ops_dag * dag)
  {
    int comm_wait_counts = 0; long long int flops_total = 0, trimming_flops = _MIN_INST_FLOPS;

    loadWorkingQueue(dag);

    for (int scheduled_insts = 0; scheduled_insts < length; scheduled_insts ++)
    {
      ready_queue * ptr;
      working_queue = working_queue -> deleteCriticalNode(&ptr, getSmallestLoad());

      int inst = ptr -> getInst(), * sync_with = ptr -> getSyncWith();
      long long int flops = ptr -> getFlops(trimming_flops), max_sync = ptr -> getMaxSync();

      const int worker_id = findWorkerWithMinimalWaiting(max_sync);

      eliminateExtraSync(sync_with, worker_id);

      for (int i = 0; i < workers; i++)
      {
        const int sync_i = sync_with[i];

        if (sync_i >= 0)
        { addWaitToWorker(sync_i, worker_id); comm_wait_counts ++; }
      }
      delete ptr;

      addInstToWorker(inst, flops, worker_id);

      updateDepsCounts(dag, inst);

      flops_total += flops;
    }

    long long flops_max_worker = 0;

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
      "Utilization Percent: %f%%. \n"
      "- Low FLOPS Instructions are trimmed.\n"
      "- Utilization may appear higher than actual.\n"
      "Avg. # of Communications per Instruction: %f. \n\n", 
      length, heaviest_worker_to_total, flops_prll, comm_per_inst);

  }

public:

  instructions_scheduler (const h_ops_dag * dag, const int num_workers_limit)
  {
    length = dag -> getLength();
    workers = num_workers_limit;
    working_queue = nullptr;
    working_queue_tail = nullptr;

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

  ~instructions_scheduler ()
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

  instructions_queue * getSchedule (const int worker_id) const
  { return (worker_id >= 0 && worker_id < workers) ? result_queues[worker_id] : nullptr; }

  int getLength (const int worker_id) const
  { return (worker_id >= 0 && worker_id < workers) ? result_queues[worker_id] -> getLength() : 0; }

  int * getLengths () const
  {
    int * lengths = new int[workers];
    for (int i = 0; i < workers; i++)
    { lengths[i] = result_queues[i] -> getLength(); }
    return lengths; 
  }

  void print () const
  {
    working_queue -> print();
    for (int i = 0; i < workers; i++)
    { printf("Worker #%d: ", i); result_queues[i] -> print(); printf("flops: %lld \n", flops_worker[i]); }
  }

};

#endif