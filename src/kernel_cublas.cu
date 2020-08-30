
#include <kernel_cublas.cuh>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

using std::vector;

void kernel_cublas(const int n_streams, const int n_insts, const vector<int>* insts, vector<double*> ptrs) {

  cudaEvent_t* comm = new cudaEvent_t[n_insts];
  for (int i = 0; i < n_insts; i++)
  { cudaEventCreate(&comm[i]); }

#pragma omp parallel for
  for (int i = 0; i < n_streams; i++) {
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cusolverDnHandle_t solvHandle;
    cublasHandle_t blasHandle;

    cusolverDnCreate(&solvHandle);
    cusolverDnSetStream(solvHandle, stream);

    double* workspace;
    cudaMalloc(&workspace, 16384);

    cublasCreate(&blasHandle);
    cublasSetStream(blasHandle, stream);
    cublasSetWorkspace(blasHandle, workspace, 16384);

    const int* pc = insts[i].data();
    int next_pc = 0;

  load:
    pc = &pc[next_pc];
    switch ((opcode_t)pc[0]) {
    case execute:
    { goto exe; }
    case signal_wait:
    { goto wait; }
    case finish: default:
    { goto fin; }
    }

  exe:
    switch ((operation_t)pc[2]) {
    case nop:
    { next_pc = nop_l; goto load; }
    case getrf: 
    {
      double* A = ptrs[pc[3]];
      int offset = pc[4], m = pc[5], n = pc[6], ld = pc[7];
      cusolverDnDgetrf(solvHandle, m, n, &A[offset], ld, workspace, nullptr, nullptr);
      cudaEventRecord(comm[pc[1]], stream);
      next_pc = 8;
      goto load;
    }
    case trsml:
    {
      double* B = ptrs[pc[3]], * L = ptrs[pc[4]];
      int offset_b = pc[5], offset_l = pc[6], n = pc[7], m = pc[8], ld_b = pc[10], ld_l = pc[11];
      bool b_T = (bool) pc[12];
      const double alpha = 1;
      cublasDtrsm(blasHandle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_UNIT, m, n, &alpha, &L[offset_l], ld_l, &B[offset_b], ld_b);
      cudaEventRecord(comm[pc[1]], stream);
      next_pc = 13;
      goto load;
    }
    /*case trsmr:
    {
      double* B = ptrs[pc[3]], * U = ptrs[pc[4]];
      int offset_b = pc[5], offset_u = pc[6], n = pc[7], m = pc[8], ld_b = pc[10], ld_u = pc[11];
      bool b_T = (bool) pc[12];
      const double alpha = 1;
      if (b_T)
      { cublasDtrsm(blasHandle, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_FULL, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, m, n, &alpha, &U[offset_u], ld_u, &B[offset_b], ld_b); }
      else
      { cublasDtrsm(blasHandle, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_FULL, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, m, n, &alpha, &U[offset_u], ld_u, &B[offset_b], ld_b); }
      cudaEventRecord(comm[pc[1]], stream);
      next_pc = 13; 
      goto load;
    }
    case gemm:
    {
      double* M = ptrs[pc[3]], * A = ptrs[pc[4]], * B = ptrs[pc[5]];
      int offset_m = pc[6], offset_a = pc[7], offset_b = pc[8], m = pc[9], n = pc[10], k = pc[11], ld_m = pc[12], ld_a = pc[13], ld_b = pc[14];
      bool a_T = (bool)pc[15], b_T = (bool)pc[16];
      double alpha = -1, beta = 1;
      cublasDgemm(blasHandle, (cublasOperation_t)a_T, (cublasOperation_t)b_T, m, n, k, &alpha, &A[offset_a], ld_a, &B[offset_b], ld_b, &beta, &M[offset_m], ld_m);
      cudaEventRecord(comm[pc[1]], stream);
      next_pc = 17; 
      goto load;
    }*/
    default:
      goto fin;

    }

  wait:
    cudaEventSynchronize(comm[pc[1]]);
    next_pc = 3;

  fin:
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    cudaFree(workspace);
    cublasDestroy(blasHandle);
    cusolverDnDestroy(solvHandle);

  }

  cudaDeviceSynchronize();

  for (int i = 0; i < n_insts; i++)
  { cudaEventDestroy(comm[i]); }
  delete[] comm;

  return;
}



