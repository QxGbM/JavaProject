#ifndef _INST_SCHEDULER_CUH
#define _INST_SCHEDULER_CUH

#include <pspl.cuh>

class inst_scheduler
{
public:
  __host__ inst_scheduler (const h_ops_dag * dag, const int num_workers_limit, const int cached_entries_limit = 0)
  {

  }
};

#endif