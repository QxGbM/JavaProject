
#ifndef _PIVOT_CUH
#define _PIVOT_CUH

#include <kernel.cuh>

template <class matrixEntriesT>
__device__ int blockAllFindRowPivot (const matrixEntriesT *matrix, const int n, const int ld)
{
  const int thread_id = thread_rank();
  const int block_size = block_dim();
  const int warp_id = warp_rank();
  const int lane_id = lane_rank();

  int index = 0;
  matrixEntriesT max = 0; 

  /* Load all row entries in warps: Each warp can handle more than 1 warpsize of data or no data. */
  for (int i = thread_id; i < n; i += block_size)
  {
    const matrixEntriesT value = abs(matrix[i * ld]);
    if (value > max)
    { max = value; index = i; }
  }

  /* Reduction in all warps. No need to explicitly sync because warp shfl implies synchronization. */
  for (int mask = warpSize / 2; mask > 0; mask /= 2) 
  {
    const matrixEntriesT s_max = __shfl_xor_sync (0xffffffff, max, mask, warpSize);
    const int s_index = __shfl_xor_sync (0xffffffff, index, mask, warpSize);
    if (s_max > max) 
    { max = s_max; index = s_index; }
  }

  __shared__ matrixEntriesT shm_max[32];
  __shared__ int shm_index[32];

  /* The first lane of each warp writes into their corresponding shared memory slot. */
  if (lane_id == 0) { shm_max[warp_id] = max; shm_index[warp_id] = index; }

  __syncthreads(); /* Sync here to make sure shared mem is properly initialized, and reductions in all warps are completed. */

  /* Do the final reduction in the first warp, if there are more than 1 warp. */
  if (block_size > warpSize && warp_id == 0) 
  {
    max = shm_max[lane_id];
    index = shm_index[lane_id];
    for (int mask = warpSize / 2; mask > 0; mask /= 2) 
    {
	  const matrixEntriesT s_max = __shfl_xor_sync (0xffffffff, max, mask, warpSize);
	  const int s_index = __shfl_xor_sync (0xffffffff, index, mask, warpSize);
      /* Uses more strict comparison to resolve ties. */
      if (s_max > max || (s_max == max && s_index < index)) 
      { max = s_max; index = s_index; }
    }

    if (lane_id == 0)
    {
      shm_max[lane_id] = max;
      shm_index[lane_id] = index;
    }
  }

  __syncthreads(); /* Sync here to stop other warps and waits for warp 0. */

  return shm_index[0];
}

template <class matrixEntriesT>
__device__ void blockSwapNSeqElements (matrixEntriesT *row1, matrixEntriesT *row2, const int n)
{
  /* Using a group of threads to exchange all elements in row with target row. */
  const int thread_id = thread_rank();
  const int block_size = block_dim();
  for (int i = thread_id; i < n; i += block_size) /* swapping n elements in two rows. */
  {
    const matrixEntriesT t = row1[i];
    row1[i] = row2[i]; 
    row2[i] = t;
  }
}

template <class matrixEntriesT>
__device__ void blockApplyPivot (matrixEntriesT *matrix, const int *pivot, const int nx, const int ny, const int ld, const bool recover = false)
{
  /* Using a group of threads to apply pivot the pivot swaps to the matrix. Recover flag retrieves original matrix. */
  for (int i = 0; i < ny; i++) 
  {
    __shared__ bool smallest_row_in_cycle;
    if (thread_rank() == 0)
    {
      smallest_row_in_cycle = true;
      int swapping_with = pivot[i];

      while (smallest_row_in_cycle && swapping_with != i)
      {
        if (swapping_with < i) { smallest_row_in_cycle = false; }
        swapping_with = pivot[swapping_with];
      }
    }
    __syncthreads();
    
    if (smallest_row_in_cycle)
    {
      int source_row = i, swapping_with = pivot[i];
      while (swapping_with != i) 
      { 
        blockSwapNSeqElements <matrixEntriesT> (&matrix[source_row * ld], &matrix[swapping_with * ld], nx);
        source_row = recover ? i : swapping_with;
        swapping_with = pivot[swapping_with];
      }
    }
  }
}

__device__ void resetPivot (int *pivot, const int n)
{
  const int thread_id = thread_rank();
  const int block_size = block_dim();
  for (int i = thread_id; i < n; i += block_size) 
  { pivot[i] = i; }
}


#endif