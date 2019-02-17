
#ifndef _CUDA_HELPER_CUH
#define _CUDA_HELPER_CUH

#define BLOCK_SIZE 16 // Each block has 16^2 = 1024 threads, make sure the cuda device allows
#define CUDA_DEVICE 0

#include <cuda.h>
#include <cooperative_groups.h>

using namespace cooperative_groups;

template <class matrixEntriesT>
__device__ void matrix_copy_inDevice (thread_group g, matrixEntriesT *target, const matrixEntriesT *source, 
  const int nx, const int ny, const int ld_target, const int ld_source)
{
  for (int i = g.thread_rank(); i < nx * ny; i += g.size())
  { 
    const int row = i / nx, col = i - row * nx;
    target[row * ld_target + col] = source[row * ld_source + col];
  }
  g.sync();
}

template <class matrixEntriesT>
__device__ void matrix_copy_inDevice (thread_group g, matrixEntriesT *target, const matrixEntriesT *source, 
  const int nx, const int ny, const int ld)
{
  matrix_copy_inDevice (g, target, source, nx, ny, ld, ld);
}

template <class matrixEntriesT>
__device__ void matrix_copy_inDevice (thread_group g, matrixEntriesT *target, const matrixEntriesT *source, 
  const int nx, const int ny)
{
  matrix_copy_inDevice (g, target, source, nx, ny, nx, nx);
}


#endif