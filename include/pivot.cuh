#ifndef PIVOT_CUH
#define PIVOT_CUH

#include <helper_functions.h>
#include <cuda_helper_functions.cuh>
#include <cooperative_groups.h>

using namespace cooperative_groups;

template <class matrixEntriesT, unsigned int tile_size>
__device__ unsigned int row_pivot (const unsigned int row, matrixEntriesT *matrix, const unsigned int nx, const unsigned int ld, const unsigned int ny)
{
  /*
  * Using 1 block, generates 1 row permutation for the input row
  * Overwrites p from index 0 to min(ny, nx) - 1. P[y, x] = 1 is transformed to p[y] = x to conserve space.
  * The input matrix is overwritten with M', where the diagonal is made as large as possible.
  * There is no limitations on the input matrix size, except that ld, the horizontal offset, needs to be not less than nx.
  */

  thread_block g = this_thread_block();
  thread_block_tile<tile_size> tile = tiled_partition<tile_size>(g);

  const unsigned int tile_id = g.thread_rank() / tile_size;
  const unsigned int num_tiles = (g.size() + tile_size - 1) / tile_size;
  const unsigned int lane_id = g.thread_rank() - tile_id * tile_size;

  unsigned int current_index = 0;
  matrixEntriesT current_max = 0.0; 

  for (unsigned int i = tile_id; i * tile_size < ny - row; i += num_tiles) /* reduction in tiles. each tile can handle more than 1 tile of data. */
  {
    unsigned int index = row + tile_id * tile_size + lane_id;
    matrixEntriesT value = (index < ny) ? abs(matrix[index * ld + row]) : 0.0;

    if (value > current_max || ((current_max - value) < 1e-10  && index < current_index))
    { current_max = value; current_index = index; }

    for (unsigned int mask = tile_size / 2; mask > 0; mask /= 2) 
    {
      matrixEntriesT shuffled_max = tile.shfl_xor(current_max, mask);
      unsigned int shuffled_index = tile.shfl_xor(current_index, mask);
      if (shuffled_max > current_max || ((current_max - shuffled_max) < 1e-10  && shuffled_index < current_index)) 
      { current_max = shuffled_max; current_index = shuffled_index; }
    }
  }

  __shared__ matrixEntriesT shm_max[tile_size];
  __shared__ unsigned int shm_index[tile_size];
  if (tile_id == 0) { shm_max[lane_id] = 0.0; shm_index[lane_id] = 0; }
  const unsigned int n = (num_tiles + tile_size - 1) / tile_size;
  const unsigned int slot = tile_id / n;
  const unsigned int turn = tile_id - slot * n;

  g.sync();

  for (unsigned int i = 0; i < n; i ++) /* crumble differnt tiles into a single tile. */
  {
    if (lane_id == 0 && i == turn) 
    {
      if (current_max > shm_max[slot] || ((shm_max[slot] - current_max) < 1e-10  &&  current_index < shm_index[slot]))
      { shm_max[slot] = current_max; shm_index[slot] = current_index; }
    }
    g.sync();
  }

  if (tile_id == 0) /* the final reduction. */
  {
    current_max = shm_max[lane_id];
    current_index = shm_index[lane_id];
    for (unsigned int mask = tile_size / 2; mask > 0; mask /= 2) 
    {
      matrixEntriesT shuffled_max = tile.shfl_xor(current_max, mask);
      unsigned int shuffled_index = tile.shfl_xor(current_index, mask);
      if (shuffled_max > current_max || ((current_max - shuffled_max) < 1e-10  && shuffled_index < current_index)) 
      { current_max = shuffled_max; current_index = shuffled_index; }
    }
    shm_max[lane_id] = current_max;
    shm_index[lane_id] = current_index;
  }

  g.sync();
  const unsigned int warp_id = g.thread_rank() / warpSize;
  const unsigned int warp_slot = warp_id - (warp_id / tile_size) * tile_size; /* warps are mapped to different slots to maximize bandwidth. */
  current_max = shm_max[warp_slot];
  current_index = shm_index[warp_slot];

  return current_index;
}

__global__ void partial_pivot_kernel2 (double *matrix, const unsigned nx, const unsigned ld, const unsigned ny, unsigned *p)
{
  row_pivot <double, 16> (4, matrix, nx, ld, ny);
}

__global__ void partial_pivot_kernel (double *matrix, const unsigned nx, const unsigned ld, const unsigned ny, unsigned *p)
{

  const unsigned x = threadIdx.x, y = threadIdx.y, b = blockIdx.x;
  const unsigned row = b * BLOCK_SIZE + y, col = b * BLOCK_SIZE + x;
  const unsigned ld_aligned = ((ld + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

  __shared__ double shm_matrix_rows[BLOCK_SIZE * BLOCK_SIZE];
  __shared__ double shm_matrix_cols[BLOCK_SIZE * BLOCK_SIZE];
  __shared__ unsigned shm_row_pref[BLOCK_SIZE * BLOCK_SIZE];
  __shared__ unsigned shm_col_pref[BLOCK_SIZE * BLOCK_SIZE];
  
  double d = (row < ny && col < nx) ? matrix[row * ld_aligned + col] : 0;
  shm_matrix_cols[y * BLOCK_SIZE + x] = shm_matrix_rows[y * BLOCK_SIZE + x] = (d >= 0) ? d : -d;
  shm_row_pref[y * BLOCK_SIZE + x] = x;
  shm_col_pref[y * BLOCK_SIZE + x] = y;

  const unsigned x2 = x * 2, y2 = y * 2, n = BLOCK_SIZE / 2 + 1;
  __syncthreads();

  for (unsigned i = 0; i < n; i++) /* sorting to get preference tables */
  {
    if (x2 + 1 < BLOCK_SIZE) /* even indexes for row preferences */
    {
      const double d0 = shm_matrix_rows[y * BLOCK_SIZE + x2];
      const double d1 = shm_matrix_rows[y * BLOCK_SIZE + (x2 + 1)];
      if (d0 < d1) /* re-order the bigger element to front */
      { 
        shm_matrix_rows[y * BLOCK_SIZE + x2] = d1; shm_matrix_rows[y * BLOCK_SIZE + (x2 + 1)] = d0;
        const unsigned u0 = shm_row_pref[y * BLOCK_SIZE + x2], u1 = shm_row_pref[y * BLOCK_SIZE + (x2 + 1)];
        shm_row_pref[y * BLOCK_SIZE + x2] = u1; shm_row_pref[y * BLOCK_SIZE + (x2 + 1)] = u0;
      }
    }
    if (y2 + 1 < BLOCK_SIZE) /* even indexes for col preferences */
    {
      const double d0 = shm_matrix_cols[y2 * BLOCK_SIZE + x];
      const double d1 = shm_matrix_cols[(y2 + 1) * BLOCK_SIZE + x];
      if (d0 < d1) /* re-order the bigger element to front */
      { 
        shm_matrix_cols[y2 * BLOCK_SIZE + x] = d1; shm_matrix_cols[(y2 + 1) * BLOCK_SIZE + x] = d0;
        const unsigned u0 = shm_col_pref[y2 * BLOCK_SIZE + x], u1 = shm_col_pref[(y2 + 1) * BLOCK_SIZE + x];
        shm_col_pref[y2 * BLOCK_SIZE + x] = u1; shm_col_pref[(y2 + 1) * BLOCK_SIZE + x] = u0;
      }
    }
    __syncthreads();

    if (x2 + 2 < BLOCK_SIZE) /* odd indexes for row preferences */
    {
      const double d0 = shm_matrix_rows[y * BLOCK_SIZE + (x2 + 1)];
      const double d1 = shm_matrix_rows[y * BLOCK_SIZE + (x2 + 2)];
      if (d0 < d1) /* re-order the bigger element to front */
      { 
        shm_matrix_rows[y * BLOCK_SIZE + (x2 + 1)] = d1; shm_matrix_rows[y * BLOCK_SIZE + (x2 + 2)] = d0;
        const unsigned u0 = shm_row_pref[y * BLOCK_SIZE + (x2 + 1)], u1 = shm_row_pref[y * BLOCK_SIZE + (x2 + 2)];
        shm_row_pref[y * BLOCK_SIZE + (x2 + 1)] = u1; shm_row_pref[y * BLOCK_SIZE + (x2 + 2)] = u0;
      }
    }
    if (y2 + 2 < BLOCK_SIZE) /* odd indexes for col preferences */
    {
      const double d0 = shm_matrix_cols[(y2 + 1) * BLOCK_SIZE + x];
      const double d1 = shm_matrix_cols[(y2 + 2) * BLOCK_SIZE + x];
      if (d0 < d1) /* re-order the bigger element to front */
      { 
        shm_matrix_cols[(y2 + 1) * BLOCK_SIZE + x] = d1; shm_matrix_cols[(y2 + 2) * BLOCK_SIZE + x] = d0;
        const unsigned u0 = shm_col_pref[(y2 + 1) * BLOCK_SIZE + x], u1 = shm_col_pref[(y2 + 2) * BLOCK_SIZE + x];
        shm_col_pref[(y2 + 1) * BLOCK_SIZE + x] = u1; shm_col_pref[(y2 + 2) * BLOCK_SIZE + x] = u0;
      }
    }
    __syncthreads();
  }

  __shared__ unsigned shm_col_pref_trans[BLOCK_SIZE * BLOCK_SIZE]; /* a magical translation of the column preference. */
  shm_col_pref_trans[x * BLOCK_SIZE + shm_col_pref[y * BLOCK_SIZE + x]] = y;

  /* Very Important Notice:
  * In the following sections, there are multiple threads mapping to the same shared memory address.
  * According to the CUDA documents, if these threads are packed in a warp, the load / save word will not serialize,
  * but change to a broadcast over the threads. If the BLOCK_SIZE is a multiple of warp size, it should do.
  */
  __shared__ unsigned shm_px[BLOCK_SIZE];
  __shared__ unsigned shm_py[BLOCK_SIZE];
  __shared__ unsigned shm_r[BLOCK_SIZE]; /* valied entries are smaller than BLOCK_SIZE, including both pairs and ranks. */
  shm_px[x] = BLOCK_SIZE; shm_py[x] = BLOCK_SIZE; shm_r[x] = BLOCK_SIZE; 
  
  __shared__ unsigned counter[BLOCK_SIZE];
  __shared__ bool not_all_paired;
  counter[x] = 0; not_all_paired = true;
  __syncthreads();

  while (not_all_paired) /* matching. */
  {
    not_all_paired = false;
    if ((shm_px[x] == BLOCK_SIZE) && (y == shm_row_pref[x * BLOCK_SIZE + counter[x]])) /* if x propose to y. */
    {
      const unsigned rank = shm_col_pref_trans[y * BLOCK_SIZE + x];
      const unsigned old_rank = atomicMin(&shm_r[y], rank);
      if (old_rank > rank) /* x is accepted by y. */
      { 
        const unsigned old_pair = atomicExch(&shm_py[y], x);
        shm_px[x] = y;
        if (old_pair < BLOCK_SIZE) /* if y has a old pair, it gets dumped. */
        { 
          shm_px[old_pair] = BLOCK_SIZE; 
          counter[old_pair]++; 
          not_all_paired = true; 
        }
      }
      else /* x is reject by y. */
      { not_all_paired = true; counter[x]++; }
    }
    __syncthreads();

  }


  /* ------------- TESTING PRINTOUTS start --------------- 

  for (unsigned i = 0; i < BLOCK_SIZE; i++) { for (unsigned j = 0; j < BLOCK_SIZE; j++) if(y == 0 && x == 0) printf("%d, ", shm_row_pref[i * BLOCK_SIZE + j]); if(y == 0 && x == 0) printf("\n"); __syncthreads(); }

  if (y == 0 && x == 0) { printf("\n"); } __syncthreads();

  for (unsigned i = 0; i < BLOCK_SIZE; i++) { for (unsigned j = 0; j < BLOCK_SIZE; j++) if(y == 0 && x == 0) printf("%d, ", shm_col_pref[i * BLOCK_SIZE + j]); if(y == 0 && x == 0) printf("\n"); __syncthreads(); }

  if (y == 0 && x == 0) { printf("\n"); } __syncthreads();

  for (unsigned i = 0; i < BLOCK_SIZE; i++) { for (unsigned j = 0; j < BLOCK_SIZE; j++) if(y == 0 && x == 0) printf("%d, ", shm_col_pref_trans[i * BLOCK_SIZE + j]); if(y == 0 && x == 0) printf("\n"); __syncthreads();}

  if (y == 0 && x == 0) { printf("\n"); } __syncthreads();

  if(y == 0) { printf("x: %d pairs with y: %d\n", x, shm_px[x]); } __syncthreads();

  if (y == 0 && x == 0) { printf("\n"); } __syncthreads();

  if(x == 0) { printf("y: %d pairs with x: %d\n", y, shm_py[y]); } __syncthreads();

   ------------- TESTING PRINTOUTS end --------------- */

  // TODO write back to the matrix;
    
}

#endif