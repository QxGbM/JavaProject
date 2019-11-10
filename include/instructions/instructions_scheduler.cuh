
#pragma once
#ifndef _INSTRUCTIONS_SCHEDULER_CUH
#define _INSTRUCTIONS_SCHEDULER_CUH

#include <definitions.cuh>
#include <h_ops/dev_hierarchical_ops_dag.cuh>

class instructions_queue
{
private:
  int inst;
  bool ex_w;
  long long int starting_flops_count;
  instructions_queue * next;

public:
  instructions_queue (const int inst_in, const bool ex_w_in, const long long int flops_in);

  ~instructions_queue ();

  int getInst () const;

  bool getExW () const;

  long long int getElapsedFlops () const;

  instructions_queue * getNext () const;

  instructions_queue * setNext (const int inst_in, const bool ex_w_in, const long long int flops_in);

  int getLength () const;

  bool * getExList (bool * list = nullptr) const;

  void print() const;

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
  ready_queue (const int inst_in, const int workers, const h_ops_dag * dag, const long long int * flops_after_inst, const int * inst_executed_by, ready_queue * next_q = nullptr);

  ~ready_queue ();

  int getInst() const;

  int getNumDeps () const;

  long long int getFlops (const long long int min_flops = _MIN_INST_FLOPS) const;

  long long int getMaxSync() const;

  int * getSyncWith() const;

  ready_queue * getLast();

  ready_queue * setNext (ready_queue * next_q);

  bool hookup (ready_queue * queue);

  ready_queue * deleteCriticalNode (ready_queue ** deleted_ptr_out, const long long int flops_synced);

  void print() const;

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

  long long int getSmallestLoad();

  void loadWorkingQueue (const h_ops_dag * dag);

  long long int findLatestSyncs (const h_ops_dag * dag, int * sync_with, const int inst);

  int findWorkerWithMinimalWaiting (const long long int max_sync);

  void eliminateExtraSync (int * sync_with, const int worker_id);

  void addInstToWorker (const int inst, const long long int flops_anticipated, const int worker_id);

  void addWaitToWorker (const int inst, const int worker_id);

  void updateDepsCounts (const h_ops_dag * dag, const int inst_finished);

  void schedule (const h_ops_dag * dag);

public:

  instructions_scheduler (const h_ops_dag * dag, const int num_workers_limit);

  ~instructions_scheduler ();

  instructions_queue * getSchedule (const int worker_id) const;

  int getLength (const int worker_id) const;

  int * getLengths () const;

  cudaError_t analyzeClocks (unsigned long long int ** clocks_gpu, const unsigned long long int block_ticks = _TICKS, const int row_blocks = _ROW_BLOCKS) const;

  void print () const;

};

#endif