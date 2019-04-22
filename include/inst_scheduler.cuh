#ifndef _INST_SCHEDULER_CUH
#define _INST_SCHEDULER_CUH

#include <pspl.cuh>

class inst_queue
{
private:
  int inst;
  int n_deps;
  bool ex_w;
  inst_queue * next;

public:
  __host__ inst_queue (const int inst_in, const int n_deps_in, const bool ex_w_in, inst_queue * next_q = nullptr)
  {
    inst = inst_in;
    n_deps = (n_deps_in > 0) ? n_deps_in : 0;
    ex_w = ex_w_in;
    next = next_q;
  }

  __host__ ~inst_queue ()
  { delete next; }

  __host__ inline int getInst () const
  { return inst; }

  __host__ inline int getNDeps () const
  { return n_deps; }

  __host__ inline bool getExW () const
  { return ex_w; }

  __host__ inline inst_queue * getNext() const
  { return next; }

  __host__ int getInst_Index (const int index) const
  {
    int i = 0;
    for (const inst_queue * ptr = this; ptr != nullptr; ptr = ptr -> next)
    { if (i == index) { return ptr -> inst; } else { i++; } }
    return -1;
  }

  __host__ int getIndex_InstExe (const int inst_in) const
  {
    int i = 0;
    for (const inst_queue * ptr = this; ptr != nullptr; ptr = ptr -> next)
    { if (ptr -> inst == inst_in && ptr -> ex_w) { return i; } else { i++; } }
    return -1;
  }

  __host__ int getNumInsts() const
  {
    int i = 0;
    for (const inst_queue * ptr = this; ptr != nullptr; ptr = ptr -> next) { i++; }
    return i;
  }

  __host__ inst_queue * removeFirst (const int index)
  { 
    int i = 0;
    for (inst_queue * ptr = this; ptr != nullptr; ptr = ptr -> next)
    {
      if (i == index - 1)
      {
        inst_queue * p = ptr -> next;
        ptr -> next = nullptr;
        delete this;
        return p;
      }
      i++;
    }
    delete this;
    return nullptr;
  }

  __host__ int hookup (const int inst_in, const int n_deps_in, const bool ex_w_in)
  {
    int i = 0;
    for (inst_queue * ptr = this; ptr != nullptr; ptr = ptr -> next)
    { 
      if (ptr -> inst == inst_in)
      { return i; }
      else if (ptr -> next == nullptr) 
      { ptr -> next = new inst_queue (inst_in, n_deps_in, ex_w_in); return i + 1; }
      else if ((ptr -> next -> n_deps) < n_deps_in)
      { ptr -> next = new inst_queue (inst_in, n_deps_in, ex_w_in, ptr -> next); return i + 1; }
      else
      { i++; }
    }
    return -1;
  }

  __host__ void print() const
  {
    for (const inst_queue * ptr = this; ptr != nullptr; ptr = ptr -> next) 
    {
      if (!(ptr -> ex_w)) { printf("w"); }
      printf("%d ", ptr -> inst);
    }
    printf("\n");
  }
};

class inst_scheduler
{
private:
  int length;
  int workers;
  inst_queue * working_queue;
  inst_queue ** result_queues;
  int * inward_deps_counter;
  int * state;

  __host__ void load_working_queue (const h_ops_dag * dag)
  {
    for (int i = 0; i < length; i++)
    {
      if (inward_deps_counter[i] == 0)
      {
        const int dep_count = dag -> getDepCount_from(i);
        if (working_queue == nullptr)
        { working_queue = new inst_queue(i, dep_count, true); }
        else if (working_queue -> getNDeps() < dep_count)
        { working_queue = new inst_queue(i, dep_count, true, working_queue); }
        else
        { working_queue -> hookup(i, dep_count, true); }
      }
    }
  }

  __host__ int add_inst (const int inst, const bool ex_w, const int worker_id)
  {
    if (result_queues[worker_id] == nullptr)
    { result_queues[worker_id] = new inst_queue(inst, 0, ex_w); return 0; }
    else
    { return result_queues[worker_id] -> hookup(inst, 0, ex_w); }
  }

  __host__ int commWriteToState (const int dep_src, const int dep_dest, const int worker_dest)
  {
    if (state[worker_dest * length + dep_src] >= 0) { return -1; }

    int worker_src, signal_src = -1;

    for (int i = 0; i < workers && signal_src == -1; i++)
    { 
      signal_src = result_queues[i] -> getIndex_InstExe (dep_src); 
      if (signal_src != -1) { worker_src = i; } 
    }

    const int signal_dest = result_queues[worker_dest] -> getNumInsts();

    for (int i = 0; i < length; i++)
    {
      const int inst_completed_src = state[worker_src * length + i], inst_completed_dest = state[worker_dest * length + i];
      if (inst_completed_src >= 0 && inst_completed_src <= signal_src && inst_completed_dest == -1)
      { state[worker_dest * length + i] = signal_dest; }
    }

    return dep_src;
  }

  __host__ void schedule (const h_ops_dag * dag)
  {
    int scheduled_insts = 0, iter = 0;
    while (scheduled_insts < length && iter < _MAX_SCHEDULER_ITERS)
    {
      load_working_queue (dag);
      for (int i = 0; i < workers; i++)
      {
        const int inst = working_queue -> getInst_Index(i);
        if (inst != -1)
        {
          for (int j = 0; j < inst; j++)
          {
            if (dag -> getDep(j, inst) > no_dep)
            { 
              const int wait = commWriteToState(j, inst, i);
              if (wait >= 0) 
              { add_inst(wait, false, i); } 
            }
          }

          state[i * length + inst] = add_inst(inst, true, i);

          for (int j = inst; j < length; j++)
          {
            if (dag -> getDep(inst, j) > no_dep)
            { inward_deps_counter[j]--; }
          }

          inward_deps_counter[inst] = -1;
          scheduled_insts++;
        }
      }
      working_queue = working_queue -> removeFirst(workers);
      iter++;
    }

    if (scheduled_insts == length)
    { printf("Successfully scheduled with %d iterations. \n", iter); }
    else
    { printf("Reached max iterations: %d. \n", iter); }
  }

public:

  __host__ inst_scheduler (const h_ops_dag * dag, const int num_workers_limit)
  {
    length = dag -> getLength();
    workers = num_workers_limit;
    working_queue = nullptr;
    result_queues = new inst_queue * [workers];
    inward_deps_counter = new int [length];
    state = new int [length * workers];

    for (int i = 0; i < workers; i++)
    { result_queues[i] = nullptr; }

    for (int i = 0; i < length; i++)
    { inward_deps_counter[i] = dag -> getDepCount_to(i); }

    memset(state, 0xffffffff, length * workers * sizeof(int));

    schedule (dag);
  }

  __host__ ~inst_scheduler ()
  {
    delete working_queue;

    for (int i = 0; i < workers; i++)
    { delete result_queues[i]; }

    delete[] result_queues;
    delete[] inward_deps_counter;
    delete[] state;
  }

  __host__ inline inst_queue * getSchedule (const int worker_id) const
  { return (worker_id >= 0 && worker_id < workers) ? result_queues[worker_id] : nullptr; }

  __host__ void print () const
  {
    for (int i = 0; i < workers; i++)
    { printf("%d: ", i); result_queues[i] -> print(); }
  }
};

#endif