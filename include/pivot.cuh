#ifndef PIVOT_CUH
#define PIVOT_CUH

#include <cooperative_groups.h>

using namespace cooperative_groups;

template <class matrixEntriesT, int tile_size>
__device__ int blockAllFindRowPivot (const int row, const matrixEntriesT *matrix, const int nx, const int ld, const int ny)
{
  /*
  * Using 1 complete thread block, finds the diagonal element with greatest absolute value for the input row.
  * All parameters are read only, which implies this will not do row exchange. There is another function to do that.
  * Output (for all threads) is the row number where matrix[output_row][row] has the largest absolute value.
  *
  * This function uses tile_size as a template to control the amount of shared mem and reduction size. 
  * Typically it should be warpSize 32, but when shared memory is not enough, tile_size can be made smaller.
  * The total amount of allocated shared memory is tile_size * (size per matrix entry + size of int).
  * Otherwise, try making tile_size as large as possible. Reduction in tiles is O(log n) scaled, while loading data in tiles is O(n).
  * If tile_size is small, the complexity of this function approches O(n).
  *
  * There is no limitations on the input matrix size, except that ld, the horizontal offset, needs to be not less than nx.
  */

  const thread_block g = this_thread_block();
  const thread_block_tile<tile_size> tile = tiled_partition<tile_size>(g);

  const int tile_id = g.thread_rank() / tile_size;
  const int num_tiles = (g.size() + tile_size - 1) / tile_size;
  const int lane_id = g.thread_rank() - tile_id * tile_size;

  int current_index = 0;
  matrixEntriesT current_max = 0; 

  /* Load all row entries in tiles: Each tile can handle more than 1 tilesize of data or no data. */
  for (int i = tile_id; i * tile_size < ny - row; i += num_tiles)
  {
    const int index = row + i * tile_size + lane_id;
    const matrixEntriesT value = (index < ny) ? abs(matrix[index * ld + row]) : 0;
    if (value > current_max || (value == current_max && index < current_index))
    { current_max = value; current_index = index; }
  }

  /* Reduction in all tiles. No need to sync because threads in a tile are always in a warp. */
  for (int mask = tile_size / 2; mask > 0; mask /= 2) 
  {
    const matrixEntriesT shuffled_max = tile.shfl_xor(current_max, mask);
    const int shuffled_index = tile.shfl_xor(current_index, mask);
    if (shuffled_max > current_max || (shuffled_max == current_max && shuffled_index < current_index)) 
    { current_max = shuffled_max; current_index = shuffled_index; }
  }

  const int n = (num_tiles + tile_size - 1) / tile_size;
  const int slot = tile_id / n;
  const int turn = tile_id - slot * n;

  __shared__ matrixEntriesT shm_max[tile_size];
  __shared__ int shm_index[tile_size];
  if (tile_id == 0) { shm_max[lane_id] = 0; shm_index[lane_id] = 0; }

  g.sync(); /* Sync here to make sure shared mem is properly initialized, and reductions in all tiles are completed. */

  /* Crumbles all tiles into a single tile. */
  for (int i = 0; i < n; i++) 
  {
    if (lane_id == 0 && i == turn) /* The first lane in each tile taking turns writing to shared mem. */
    {
      const int index = shm_index[slot];
      const matrixEntriesT value = shm_max[slot];
      if (current_max > value || (current_max == value && current_index < index))
      { shm_max[slot] = current_max; shm_index[slot] = current_index; }
    }
    g.sync(); /* Sync here to make sure all tiles are at the same turn. */
  }

  /* The final reduction in the first tile. */
  if (tile_id == 0) 
  {
    current_max = shm_max[lane_id];
    current_index = shm_index[lane_id];
    for (int mask = tile_size / 2; mask > 0; mask /= 2) 
    {
      const matrixEntriesT shuffled_max = tile.shfl_xor(current_max, mask);
      const int shuffled_index = tile.shfl_xor(current_index, mask);
      if (shuffled_max > current_max || (shuffled_max == current_max && shuffled_index < current_index)) 
      { current_max = shuffled_max; current_index = shuffled_index; }
    }
    shm_max[lane_id] = current_max;
    shm_index[lane_id] = current_index;
  }

  g.sync(); /* Sync here to make sure all threads are reading correct values. */
  const int warp_id = g.thread_rank() / warpSize;
  const int warp_slot = warp_id - (warp_id / tile_size) * tile_size; /* warps are mapped to different slots to maximize reading bandwidth. */
  current_max = shm_max[warp_slot];
  current_index = shm_index[warp_slot];

  return current_index;
}

template <class matrixEntriesT>
__device__ void blockExchangeRow (thread_group g, const int row, const int target, int *pivot, matrixEntriesT *matrix, 
  const int nx, const int ld, const int ny)
{
  /* Using a group of threads to exchange all elements in row with target row. */
  if (row < ny && target < ny)
  {
    for (int i = g.thread_rank(); i < nx; i += g.size()) /* swapping elements in the matrix. */
    {
      const matrixEntriesT t1 = matrix[row * ld + i], t2 = matrix[target * ld + i];
      matrix[row * ld + i] = t2; matrix[target * ld + i] = t1;
    }
    if (pivot != nullptr && g.thread_rank() == 0) /* swapping the row numbers in pivot. */
    { const int p = pivot[row]; pivot[row] = pivot[target]; pivot[target] = p; }
    g.sync();
  }
}

template <class matrixEntriesT>
__device__ void blockApplyPivot (thread_group g, const int *pivot, const bool recover, matrixEntriesT *matrix, 
  const int nx, const int ld, const int ny)
{
  /* Using a group of threads to apply pivot the pivot swaps to the matrix. Recover flag retrieves original matrix. */
  for (int i = 0; i < ny; i++) 
  {
    bool smallest_row_in_cycle = true;
    int swapping_with = pivot[i];
    while (smallest_row_in_cycle && swapping_with != i)
    {
      if (swapping_with < i) { smallest_row_in_cycle = false; }
      swapping_with = pivot[swapping_with];
    }

    if (smallest_row_in_cycle)
    {
      int source_row = i;
      swapping_with = pivot[i];
      while (swapping_with != i) 
      { 
        blockExchangeRow(g, source_row, swapping_with, nullptr, matrix, nx, ld, ny);
        source_row = recover ? i : swapping_with;
        swapping_with = pivot[swapping_with];
      }
    }
  }
}


#endif