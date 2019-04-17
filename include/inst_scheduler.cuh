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
  __host__ inst_queue (const int inst_in, const int n_deps_in, const bool ex_w_in)
  {
    inst = inst_in;
    n_deps = (n_deps_in > 0) ? n_deps_in : 0;
    ex_w = ex_w_in;
    next = nullptr;
  }

  __host__ ~inst_queue ()
  {
    delete next; 
  }

  __host__ int get_inst (const int index)
  {
    int i = 0;
    for (inst_queue * ptr = this; ptr != nullptr; ptr = ptr -> next)
    {
      if (i == index) { return ptr -> inst; }
      i++;
    }
    return -1;
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

  __host__ void hookup (const int inst_in, const int n_deps_in, const bool ex_w_in)
  {
    for (inst_queue * ptr = this; ptr != nullptr; ptr = ptr -> next)
    { 
      if (ptr -> inst == inst_in)
      {
        return;
      }
      else if (ptr -> next == nullptr) 
      { 
        ptr -> next = new inst_queue (inst_in, n_deps_in, ex_w_in); 
        return; 
      }
      else if ((ptr -> next -> n_deps) < n_deps_in)
      {
        inst_queue * p = new inst_queue (inst_in, n_deps_in, ex_w_in);
        p -> next = ptr -> next;
        ptr -> next = p;
        return;
      }
    }
  }

  __host__ int getLaterInst (const int original, const int inst_in) const
  {
    bool original_ex = false, inst_in_ex = false;

    for (const inst_queue * ptr = this; ptr != nullptr; ptr = ptr -> next)
    {
      if (ptr -> inst == original && ptr -> ex_w)
      {
        original_ex = true; 
        if (inst_in_ex) 
        { return original; } 
      }
      else if (ptr -> inst == inst_in && ptr -> ex_w)
      {
        inst_in_ex = true;
        if (original_ex)
        { return inst_in; }
      }
    }

    if (inst_in_ex && !original_ex) 
    { return inst_in; }
    else if (!inst_in_ex && original_ex)
    { return original; }
    else
    { return -1; }
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

  __host__ void load_working_queue (const h_ops_dag * dag)
  {
    for (int i = 0; i < length; i++)
    {
      if (inward_deps_counter[i] == 0)
      {
        if (working_queue == nullptr)
        { working_queue = new inst_queue(i, dag -> getDepCount_from(i), true); }
        else
        { working_queue -> hookup(i, dag -> getDepCount_from(i), true); }
      }
    }
  }

  __host__ void add_inst (const int inst, const bool ex_w, const int worker_id)
  {
    if (result_queues[worker_id] == nullptr)
    { result_queues[worker_id] = new inst_queue(inst, 0, ex_w); }
    else
    { result_queues[worker_id] -> hookup(inst, 0, ex_w); }
  }

  __host__ int * wait_list (const h_ops_dag * dag, const int inst)
  {
    int * list = new int[workers];
    memset(list, -1, workers * sizeof(int));

    for (int i = 0; i < inst; i++)
    {
      if ((dag -> getDep(i, inst)) > no_dep)
      {
        for (int j = 0; j < workers; j++)
        { list[j] = result_queues[j] -> getLaterInst(list[j], i); }
      }
    }

    return list;
  }

  __host__ void schedule (const h_ops_dag * dag)
  {
    int scheduled_insts = 0, iter = 0;
    while (scheduled_insts < length && iter < 100)
    {
      load_working_queue(dag);
      for (int i = 0; i < workers; i++)
      {
        int inst = working_queue -> get_inst(i);
        if (inst != -1)
        {
          int * list = wait_list (dag, inst);
          for (int j = 0; j < workers; j++)
          { if (list[j] != -1) add_inst (list[j], false, i); }
          delete list;
          add_inst (inst, true, i);

          list = dag -> flattenDep_from(inst);
          for (int j = 0; j < dag -> getDepCount_from(inst); j++)
          { inward_deps_counter[list[j]]--; }
          delete list;
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

    for (int i = 0; i < workers; i++)
    { result_queues[i] = nullptr; }

    for (int i = 0; i < length; i++)
    { inward_deps_counter[i] = dag -> getDepCount_to(i); }

    schedule(dag);
  }

  __host__ ~inst_scheduler ()
  {
    delete working_queue;

    for (int i = 0; i < workers; i++)
    { delete result_queues[i]; }

    delete[] result_queues;
    delete[] inward_deps_counter;
  }

  __host__ void print () const
  {

    for (int i = 0; i < workers; i++)
    { 
      printf("%d: ", i);
      result_queues[i] -> print(); 
    }
  }
};

#endif