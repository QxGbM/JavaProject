
#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <cuda.h>

#include <helper_functions.h>

#define BLOCK_SIZE 16 // Each block has 32^2 = 1024 threads, make sure the cuda device allows
#define CUDA_DEVICE 0

extern "C" void test_dense_getrf_1x1 ();

extern "C" void test_dense_getrf_nx1 (const unsigned ny);

extern "C" void test_dense_getrf_1xn (const unsigned nx);

extern "C" void test_dense_getrf_nxn (const unsigned nx, const unsigned ny);

extern "C" void test_inverse (const unsigned nx, const unsigned ny);

extern "C" void test_dense_gemm_1block (const unsigned nx, const unsigned ny);

