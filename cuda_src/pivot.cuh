#include <stdio.h>
#include <cuda.h>

#include <helper_functions.h>
#include <cuda_helper_functions.cuh>

__global__ void partial_pivot_kernel (double *matrix, const unsigned nx, const unsigned ld, const unsigned ny, unsigned *p);

__host__ int main();