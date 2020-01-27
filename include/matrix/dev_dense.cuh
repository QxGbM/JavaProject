
#pragma once
#ifndef _DEV_DENSE_CUH
#define _DEV_DENSE_CUH

#include <definitions.cuh>

class dev_dense 
{
private:
  int device_id;

  int nx;
  int ny;
  int ld;

  real_t * elements;

  bool pivoted;
  int * pivot;

  int shadow_rank;
  real_t * shadow_u;
  real_t * shadow_vt;

public:

  dev_dense (const int nx_in = 0, const int ny_in = 0, const int ld_in = 0, const int shadow_rank_in = 0, const int device_id_in = 0, const bool alloc_pivot = false);

  ~dev_dense ();

  int getNx () const;

  int getNy () const;

  int getLd () const;

  real_t * getElements (const int offset = 0) const;

  int * getPivot (const int offset = 0) const;

  int getShadowRank () const;

  real_t * getShadow_U (const int offset = 0) const;

  real_t * getShadow_VT (const int offset = 0) const;

  cudaError_t resize (const int ld_in, const int ny_in);

  cudaError_t resizeColumn (const int ld_in);

  cudaError_t resizeRow (const int ny_in);

  cudaError_t resizeShadow (const int shadow_rank_in);

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

  static dev_dense * readStructureFromFile (FILE * stream, const int shadow_rank = _SHADOW_RANK);

  static dev_dense * readFromFile (const char * file_name, const int shadow_rank = _SHADOW_RANK, const bool reverse_bytes = true);
   
  void print (const int y_start = 0, const int ny_in = _PRINT_NUM_ROWS, const int x_start = 0, const int nx_in = _PRINT_NUMS_PER_ROW) const;

  real_t sqrSum() const;

  real_t L2Error (const dev_dense * matrix) const;

};


#endif