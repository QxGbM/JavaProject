
#ifndef _CUDA_HELPER_CUH
#define _CUDA_HELPER_CUH

#include <stdio.h>
#include <string.h>
#include <cuda.h>

#define BLOCK_SIZE 16 // Each block has 16^2 = 1024 threads, make sure the cuda device allows
#define CUDA_DEVICE 0

template <class matrixEntriesT>
__host__ cudaError_t matrix_copy_toDevice_sync (matrixEntriesT *matrix, matrixEntriesT **dev_matrix,
  const unsigned int nx, const unsigned int ld, const unsigned int ny, const bool use_same_offset_on_device = false)
{
  /* 
  * A synchronous copy of matrix to dev_matrix.
  * Matrix can be stored in pageable memory.
  * This function also allocates dev_matrix if it is not allocated.
  */

  if (ld < nx) { printf("MEM COPY ABORT: Matrix x's horizontal offset is less than the number of entries.\n");  return cudaErrorInvalidConfiguration; }

  cudaError_t error = cudaSuccess;
  const unsigned int offset = use_same_offset_on_device ? ld : nx;
  
  if (*dev_matrix == 0) 
  {
    error = cudaMalloc((void**)dev_matrix, offset * ny * sizeof(matrixEntriesT));
    if (error != cudaSuccess) { return error; }

    printf("Allocated %d x %d matrix in cuda global memory.\n\n", ny, offset);
  }

  for (unsigned int i = 0; i < ny; i++)
  {
    error = cudaMemcpy(&(*dev_matrix)[i * offset], &matrix[i * ld], nx * sizeof(matrixEntriesT), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) { return error; }
  }
  
  printf("Copied %d x %d entries from host to cuda global memory.\n\n", ny, nx);
  return cudaSuccess;
}

template <class matrixEntriesT>
__host__ cudaError_t matrix_copy_toHost_sync (matrixEntriesT **dev_matrix, matrixEntriesT *matrix,
  const unsigned int nx, const unsigned int ld, const unsigned int ny, const bool free_device = false, const bool use_same_offset_on_device = false)
{
  /* 
  * A synchronous copy of dev_matrix to matrix
  * Matrix can be stored in pageable memory
  */
  if (ld < nx) { printf("MEM COPY ABORT: Matrix x's horizontal offset is less than the number of entries.\n");  return cudaErrorInvalidConfiguration; }

  cudaError_t error = cudaSuccess;
  const unsigned int offset = use_same_offset_on_device ? ld : nx;

  for (unsigned int i = 0; i < ny; i++)
  {
    error = cudaMemcpy(&matrix[i * ld], &(*dev_matrix)[i * offset], nx * sizeof(matrixEntriesT), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) { return error; }
  }
  
  printf("Copied %d x %d entries from cuda global memory to host.\n\n", ny, nx);

  if (free_device)
  {
    error = cudaFree(*dev_matrix);
    if (error != cudaSuccess) { return error; }

    printf("Freed %d x %d matrix in cuda global memory.\n\n", ny, offset);
  }

  return cudaSuccess;
}

template <class matrixEntriesT>
__device__ void matrix_copy_inDevice (matrixEntriesT *target, matrixEntriesT *source, )


#endif