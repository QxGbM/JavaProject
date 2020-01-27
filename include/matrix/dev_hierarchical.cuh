
#pragma once
#ifndef _DEV_HIERARCHICAL_CUH
#define _DEV_HIERARCHICAL_CUH

#include <definitions.cuh>

class dev_hierarchical 
{
private:

  int nx;
  int * x_offsets;

  int ny;
  int * y_offsets;

  dev_h_element * elements;

public:
  
  dev_hierarchical (const int nx_in, const int ny_in, const int abs_y = 0, const int abs_x = 0, element_t type = empty, void ** elements_in = nullptr);

  ~dev_hierarchical ();

  int getNx_blocks () const;

  int getNy_blocks () const;

  int getNx_abs () const;

  int getNy_abs () const;

  bool updateOffsets (const int abs_y = 0, const int abs_x = 0);

  void setElement (void * M, const element_t type, const int x, const int y);

  dev_h_element * getElement_blocks (const int y, const int x) const;

  real_t getElement_abs (const int y_in, const int x_in) const;

  void getElement_loc (int * offset_y, int * offset_x, int * block_y, int * block_x) const;

  void getOffsets_x (int ** x) const;

  void getOffsets_y (int ** y) const;

  dev_dense * convertToDense() const;

  h_index * getRootIndex () const;

  h_ops_tree * generateOps_GETRF (const h_index * self, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_TRSML (const h_index * self, const dev_dense * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_TRSML (const h_index * self, const dev_low_rank * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_TRSML (const h_index * self, const dev_hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_TRSML (const h_index * self, const dev_h_element * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_TRSMR (const h_index * self, const dev_dense * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_TRSMR (const h_index * self, const dev_low_rank * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_TRSMR (const h_index * self, const dev_hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_TRSMR (const h_index * self, const dev_h_element * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_ACCM (const h_index * self, const h_index * index_tmp_lr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const dev_dense * A, const h_index * index_a, const dev_dense * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const dev_low_rank * A, const h_index * index_a, const dev_dense * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const dev_hierarchical * A, const h_index * index_a, const dev_dense * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const dev_h_element * A, const h_index * index_a, const dev_dense * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const dev_dense * A, const h_index * index_a, const dev_low_rank * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const dev_low_rank * A, const h_index * index_a, const dev_low_rank * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const dev_hierarchical * A, const h_index * index_a, const dev_low_rank * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const dev_h_element * A, const h_index *index_a, const dev_low_rank * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const dev_dense * A, const h_index * index_a, const dev_hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const dev_low_rank * A, const h_index * index_a, const dev_hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const dev_hierarchical * A, const h_index * index_a, const dev_hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const dev_h_element * A, const h_index * index_a, const dev_hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const dev_dense * A, const h_index * index_a, const dev_h_element * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const dev_low_rank * A, const h_index * index_a, const dev_h_element * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const dev_hierarchical * A, const h_index * index_a, const dev_h_element * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const dev_h_element * A, const h_index * index_a, const dev_h_element * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  cudaError_t loadBinary (FILE * stream, const bool reverse_bytes = true);

  static dev_hierarchical * readStructureFromFile (FILE * stream, const int shadow_rank = _SHADOW_RANK);

  static dev_hierarchical * readFromFile (const char * file_name, const int shadow_rank = _SHADOW_RANK, const bool reverse_bytes = true);

  void print (std :: vector <int> &indices) const;

  void print () const;


};

#endif