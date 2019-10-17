
#pragma once
#ifndef _DEV_HIERARCHICAL_ELEMENT_CUH
#define _DEV_HIERARCHICAL_ELEMENT_CUH

#include <pspl.cuh>

class dev_h_element 
{
private:

  void * element;
  element_t type;

public:
  
  dev_h_element (void *element_in = nullptr, const element_t type_in = empty);

  ~dev_h_element ();

  dev_dense * getElementDense() const;

  dev_low_rank * getElementLowRank() const;

  dev_hierarchical * getElementHierarchical() const;

  element_t getType() const;

  int getNx() const;

  int getNy() const;

  int getLd() const;

  int getRank() const;

  void setElement (void * element_in, element_t type_in);

  real_t getElement (const int y_in, const int x_in) const;

  dev_dense * convertToDense() const;

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

  h_ops_tree * generateOps_GEMM (const h_index * self, const dev_dense * A, const h_index * index_a, const dev_low_rank * B, const h_index * index_b, dev_temp * tmp_mngr) const;
  
  h_ops_tree * generateOps_GEMM (const h_index * self, const dev_dense * A, const h_index * index_a, const dev_hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const dev_dense * A, const h_index * index_a, const dev_h_element * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const dev_low_rank * A, const h_index * index_a, const dev_dense * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const dev_low_rank * A, const h_index * index_a, const dev_low_rank * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const dev_low_rank * A, const h_index * index_a, const dev_hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const dev_low_rank * A, const h_index * index_a, const dev_h_element * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const dev_hierarchical * A, const h_index * index_a, const dev_dense * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const dev_hierarchical * A, const h_index * index_a, const dev_low_rank * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const dev_hierarchical * A, const h_index * index_a, const dev_hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const dev_hierarchical * A, const h_index * index_a, const dev_h_element * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const dev_h_element * A, const h_index * index_a, const dev_dense * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const dev_h_element * A, const h_index * index_a, const dev_low_rank * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const dev_h_element * A, const h_index * index_a, const dev_hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const dev_h_element * A, const h_index * index_a, const dev_h_element * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  cudaError_t loadBinary (FILE * stream, const bool reverse_bytes = true);

  static void * readStructureFromFile (FILE * stream, element_t * type, const int shadow_rank = _SHADOW_RANK);

  void print(const h_index* index) const;

};


#endif