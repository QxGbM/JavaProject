
#pragma once
#ifndef _LowRank_CUH
#define _LowRank_CUH

#include <definitions.cuh>

class LowRank 
{
private:
  int nx;
  int ny;
  int rank;
  Dense * UxS;
  Dense * VT;

public:

  LowRank (const int x, const int y, const int rank_in = -1);

  LowRank (Dense * data_in);

  ~LowRank ();

  int getNx () const;

  int getNy () const;

  int getRank () const;

  Dense * getUxS () const;

  Dense * getVT () const;

  real_t * getElements (const int offset = 0) const;

  real_t getElement (const int y, const int x) const;

  Dense * convertToDense() const;

  cudaError_t adjustRank (const int rank_in);

  static h_ops_tree * generateOps_GETRF (const h_index * self, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_TRSML (const h_index * self, const Dense * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_TRSML (const h_index * self, const LowRank * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_TRSML (const h_index * self, const Hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_TRSML (const h_index * self, const Element * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_TRSMR (const h_index * self, const Dense * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_TRSMR (const h_index * self, const LowRank * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_TRSMR (const h_index * self, const Hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_TRSMR (const h_index * self, const Element * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_ACCM (const h_index * self, const h_index * index_tmp_lr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const Dense * A, const h_index * index_a, const Dense * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const LowRank * A, const h_index * index_a, const Dense * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const Hierarchical * A, const h_index * index_a, const Dense * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const Element * A, const h_index * index_a, const Dense * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const Dense * A, const h_index * index_a, const LowRank * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const LowRank * A, const h_index * index_a, const LowRank * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const Hierarchical * A, const h_index * index_a, const LowRank * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const Element * A, const h_index * index_a, const LowRank * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const Dense * A, const h_index * index_a, const Hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const LowRank * A, const h_index * index_a, const Hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const Hierarchical * A, const h_index * index_a, const Hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const Element * A, const h_index * index_a, const Hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const Dense * A, const h_index * index_a, const Element * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const LowRank * A, const h_index * index_a, const Element * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const Hierarchical * A, const h_index * index_a, const Element * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const Element * A, const h_index * index_a, const Element * B, const h_index * index_b, dev_temp * tmp_mngr);

  cudaError_t loadBinary (FILE * stream, const bool reverse_bytes = true);

  static LowRank * readStructureFromFile (FILE * stream);

  static LowRank * readFromFile (const char * file_name, const bool reverse_bytes = true);

  void print(const int y_start = 0, const int ny_in = _PRINT_NUM_ROWS, const int x_start = 0, const int nx_in = _PRINT_NUM_ROWS, const int rank_in = _PRINT_NUMS_PER_ROW) const;

};


#endif
