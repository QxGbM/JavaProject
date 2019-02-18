#ifndef _DAG_CUH
#define _DAG_CUH

#include <dev_hierarchical.cuh>

enum matrix_op_t {
  nop,
  getrf,
  getrf_pivot,
  apply_pivot,
  trsm,
  gemm,
};

struct op {
  matrix_op_t op_type;
  int *x;
  int *y;
}

struct dag {

  int n_blocks;
  int block_size;
  int ops_length;

  matrix_op_t *ops;

};

#endif