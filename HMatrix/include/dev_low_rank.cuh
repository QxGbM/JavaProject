
#pragma once
#ifndef _DEV_LOW_RANK_CUH
#define _DEV_LOW_RANK_CUH

#include <definitions.cuh>

class dev_low_rank 
{
private:
  int nx;
  int ny;
  int rank;
  dev_dense * UxS;
  dev_dense * VT;

public:

  dev_low_rank (const int x, const int y, const int rank_in = -1);

  dev_low_rank (dev_dense * data_in);

  ~dev_low_rank ();

  int getNx () const;

  int getNy () const;

  int getRank () const;

  dev_dense * getUxS () const;

  dev_dense * getVT () const;

  real_t * getElements (const int offset = 0) const;

  real_t getElement (const int y, const int x) const;

  dev_dense * convertToDense() const;

  cudaError_t adjustRank (const int rank_in);

  static h_ops_tree * generateOps_GETRF (const h_index * self, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_TRSML (const h_index * self, const dev_dense * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_TRSML (const h_index * self, const dev_low_rank * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_TRSML (const h_index * self, const dev_hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_TRSML (const h_index * self, const dev_h_element * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_TRSMR (const h_index * self, const dev_dense * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_TRSMR (const h_index * self, const dev_low_rank * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_TRSMR (const h_index * self, const dev_hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_TRSMR (const h_index * self, const dev_h_element * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_ACCM (const h_index * self, const h_index * index_tmp_lr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const dev_dense * A, const h_index * index_a, const dev_dense * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const dev_low_rank * A, const h_index * index_a, const dev_dense * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const dev_hierarchical * A, const h_index * index_a, const dev_dense * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const dev_h_element * A, const h_index * index_a, const dev_dense * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const dev_dense * A, const h_index * index_a, const dev_low_rank * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const dev_low_rank * A, const h_index * index_a, const dev_low_rank * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const dev_hierarchical * A, const h_index * index_a, const dev_low_rank * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const dev_h_element * A, const h_index * index_a, const dev_low_rank * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const dev_dense * A, const h_index * index_a, const dev_hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const dev_low_rank * A, const h_index * index_a, const dev_hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const dev_hierarchical * A, const h_index * index_a, const dev_hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const dev_h_element * A, const h_index * index_a, const dev_hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const dev_dense * A, const h_index * index_a, const dev_h_element * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const dev_low_rank * A, const h_index * index_a, const dev_h_element * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const dev_hierarchical * A, const h_index * index_a, const dev_h_element * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const dev_h_element * A, const h_index * index_a, const dev_h_element * B, const h_index * index_b, dev_temp * tmp_mngr);

  cudaError_t loadBinary (FILE * stream, const bool reverse_bytes = true);

  static dev_low_rank * readStructureFromFile (FILE * stream);

  static dev_low_rank * readFromFile (const char * file_name, const bool reverse_bytes = true);

  void print(const int y_start = 0, const int ny_in = _PRINT_NUM_ROWS, const int x_start = 0, const int nx_in = _PRINT_NUM_ROWS, const int rank_in = _PRINT_NUMS_PER_ROW) const;

};


#endif
