
#include <stdio.h>
#include <cuda.h>

#include <helper_functions.h>
#include <cuda_helper_functions.cuh>

__global__ void dense_getrf_kernel (double *matrix, const unsigned nx, const unsigned ld, const unsigned ny);

__global__ void dense_trsm_left_kernel (double *X_B, double *A, const unsigned nx_x, const unsigned nx_a, const unsigned ld_x, const unsigned ld_a, 
  const unsigned ny_ab, const double alpha, const bool unit_triangular, const bool use_lower, const bool use_upper);

__global__ void dense_trsm_right_kernel (double *X_B, double *A, const unsigned nx_ab, const unsigned ld_x, const unsigned ld_a, const unsigned ny_x, 
  const unsigned ny_a, const double alpha, const bool unit_triangular, const bool use_lower, const bool use_upper);

__global__ void dense_gemm_kernel (double *X_C, double *A, double *B, const unsigned nx_bx, const unsigned nx_a_ny_b, const unsigned ny_ax, 
  const unsigned ld_x, const unsigned ld_a, const unsigned ld_b, const double alpha, const double beta, const unsigned b2);

__host__ int dense_getrf_sync (double *matrix, const unsigned nx, const unsigned ld, const unsigned ny);

__host__ int dense_trsm_sync (double *matrix_x, const unsigned nx_x, const unsigned ld_x, const unsigned ny_x,
  double *matrix_a, const unsigned nx_a, const unsigned ld_a, const unsigned ny_a, const double alpha, const bool side, 
  const bool unit_triangular, const bool use_lower, const bool use_upper);

__host__ int dense_gemm_sync (double *matrix_x, const unsigned nx_x, const unsigned ld_x, const unsigned ny_x,
  double *matrix_a, const unsigned nx_a, const unsigned ld_a, const unsigned ny_a, 
  double *matrix_b, const unsigned nx_b, const unsigned ld_b, const unsigned ny_b, const double alpha, const double beta);

__host__ int dense_getrf_sync (Matrix *m);

__host__ int dense_trsm_sync (Matrix *x_B, Matrix *A, const double alpha, const bool side, 
  const bool unit_triangular, const bool use_lower, const bool use_upper);

__host__ int dense_gemm_sync (Matrix *x_C, Matrix *A, Matrix *B, const double alpha, const double beta);

