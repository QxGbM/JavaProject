
#pragma once
#ifndef _dev_hierarchical_OPS_CUH
#define _dev_hierarchical_OPS_CUH

#include <definitions.cuh>

class h_ops
{
protected:

  operation_t op_type;

  int n_rw;
  h_index * read_and_write;

  int n_ro;
  h_index * read_only;

  long long int flops;

public:

  h_ops ();

  h_ops (const operation_t op_in, const h_index * M);

  h_ops (const operation_t op_in, const h_index * M1, const h_index * M2);

  h_ops (const operation_t op_in, const h_index * M1, const h_index * M2, const h_index * M3);

  ~h_ops ();

  operation_t opType() const;

  dependency_t checkDependencyFrom (const h_ops * op_from) const;

  dependency_t checkDependencyTo (const h_ops * op_to) const;

  int getDataPointers (void ** data_ptrs, void ** tmp_ptrs) const;

  int writeOpParametersTo (int * inst, int * tmp_size, int * rnd_size, const int * mapping) const;

  long long int getFlops (long long int * trim);

  void getAbs_rw (const int index, int * abs_x_out, int * abs_y_out, int * nx_out, int * ny_out);

  void getAbs_ro (const int index, int * abs_x_out, int * abs_y_out, int * nx_out, int * ny_out);

  static int getTmpSize_ACCM_LR (int * offset1, int * offset2, const int nx, const int ny, const int rank1);

  static int getControl_GEMM_3x (int * t_size, const int m, const int n, const int k, const int l);

  static int getControl_GEMM_4x (int * t_size, int * offset, const int m, const int n, const int k, const int l, const int o);

  static long long int getFlops_GETRF (long long int * trim, const long long int nx, const long long int ny, const long long int trim_dim = _BLOCK_M);

  static long long int getFlops_TRSML (long long int * trim, const long long int nx_b, const long long int ny_b, const long long int nx_l, const long long int trim_dim = _BLOCK_M);

  static long long int getFlops_TRSMR (long long int * trim, const long long int nx_b, const long long int ny_b, const long long int ny_u, const long long int trim_dim = _BLOCK_M);

  static long long int getFlops_GEMM (long long int * trim, const long long int m, const long long int n, const long long int k, const long long int trim_dim = _BLOCK_M);

  static long long int getFlops_GEMM_3x (long long int * trim, const long long int m, const long long int n, const long long int k, const long long int l, const long long int trim_dim = _BLOCK_M);

  static long long int getFlops_GEMM_4x (long long int * trim, const long long int m, const long long int n, const long long int k, const long long int l, const long long int o, const long long int trim_dim = _BLOCK_M);

  static long long int getFlops_QR (long long int * trim, const long long int nx, const long long int ny, const long long int trim_dim = _BLOCK_M);

  static long long int getFlops_LrAccum (long long int * trim, const long long int nx, const long long int ny, const long long int rank1, const long long int rank2, const long long int trim_dim = _BLOCK_M);

  void print() const;
  
};

#endif