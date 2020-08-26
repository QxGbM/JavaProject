
#pragma once
#ifndef _dev_hierarchical_CUH
#define _dev_hierarchical_CUH

#include <definitions.cuh>

class Hierarchical 
{
private:

  int nx;
  int * x_offsets;

  int ny;
  int * y_offsets;

  Element * elements;

public:
  
  Hierarchical (const int nx_in, const int ny_in, const int abs_x = 0, const int abs_y = 0, element_t type = empty, void ** elements_in = nullptr);

  ~Hierarchical ();

  int getNx_blocks () const;

  int getNy_blocks () const;

  int getNx_abs () const;

  int getNy_abs () const;

  bool updateOffsets (const int abs_x = 0, const int abs_y = 0);

  void setElement (void * M, const element_t type, const int x, const int y, const int abs_x, const int abs_y);

  Element * getElement_blocks (const int y, const int x) const;

  real_t getElement_abs (const int y_in, const int x_in) const;

  void getElement_loc (int * offset_y, int * offset_x, int * block_y, int * block_x) const;

  void getOffsets_x (int ** x) const;

  void getOffsets_y (int ** y) const;

  Dense * convertToDense() const;

  h_index * getRootIndex () const;

  h_ops_tree * generateOps_GETRF (const h_index * self, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_TRSML (const h_index * self, const Dense * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_TRSML (const h_index * self, const LowRank * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_TRSML (const h_index * self, const Hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_TRSML (const h_index * self, const Element * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_TRSMR (const h_index * self, const Dense * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_TRSMR (const h_index * self, const LowRank * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_TRSMR (const h_index * self, const Hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_TRSMR (const h_index * self, const Element * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_ACCM (const h_index * self, const h_index * index_tmp_lr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const Dense * A, const h_index * index_a, const Dense * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const LowRank * A, const h_index * index_a, const Dense * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const Hierarchical * A, const h_index * index_a, const Dense * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const Element * A, const h_index * index_a, const Dense * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const Dense * A, const h_index * index_a, const LowRank * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const LowRank * A, const h_index * index_a, const LowRank * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const Hierarchical * A, const h_index * index_a, const LowRank * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const Element * A, const h_index *index_a, const LowRank * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const Dense * A, const h_index * index_a, const Hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const LowRank * A, const h_index * index_a, const Hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const Hierarchical * A, const h_index * index_a, const Hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const Element * A, const h_index * index_a, const Hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const Dense * A, const h_index * index_a, const Element * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const LowRank * A, const h_index * index_a, const Element * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const Hierarchical * A, const h_index * index_a, const Element * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  h_ops_tree * generateOps_GEMM (const h_index * self, const Element * A, const h_index * index_a, const Element * B, const h_index * index_b, dev_temp * tmp_mngr) const;

  cudaError_t loadBinary (FILE * stream, const bool reverse_bytes = true);

  static Hierarchical * readStructureFromFile (FILE * stream, const int shadow_rank = _SHADOW_RANK);

  static Hierarchical * readFromFile (const char * file_name, const int shadow_rank = _SHADOW_RANK, const bool reverse_bytes = true);

  void print (std :: vector <int> &indices) const;

  void print () const;


};

#endif