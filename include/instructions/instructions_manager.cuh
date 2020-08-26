
#pragma once
#ifndef _INSTRUCTIONS_MANAGER_CUH
#define _INSTRUCTIONS_MANAGER_CUH

#include <definitions.cuh>

using std::vector;

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

  void changeInstsSize (const int worker_id, const int size_in);

  void changePointersSize (const int size_in);

  void loadPointers (void ** ptrs_in, const int n_ptrs, int * mapping);

  int loadInsts (int * tmp_size, int * rnd_size, const int worker_id, const instructions_queue * queue, const h_ops_dag * dag, void ** tmp_ptrs, const double gpu_clock_multiplier = _CLOCK_MULTIPLIER);

public:

  instructions_manager (const int num_workers, const h_ops_dag * dag, const instructions_scheduler * schedule, void ** tmp_ptrs);

  ~instructions_manager ();

  cudaError_t getLaunchArgs (int *** dev_insts, void *** dev_ptrs, int ** comm_space, real_t*** block_tmps, real_t ** dev_rnd_seed, const unsigned int seed_in = 0) const;

  void getLaunchArgsCublas(int &n_streams, int &n_insts, vector<int>* &insts, vector<double*> &ptrs) const;

  void print (const int limit = 32, const int ptrs_limit = 32) const;

};

#endif