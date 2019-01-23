#ifndef PIVOT_CUH
#define PIVOT_CUH

#include <stdio.h>
#include <cuda.h>

#include <helper_functions.h>
#include <cuda_helper_functions.cuh>
#include <gpu_lu.cuh>

#include <cub/cub.cuh>

__global__ void partial_pivot_kernel (double *matrix, const unsigned nx, const unsigned ld, const unsigned ny, unsigned *p)
{
  /*
  * Generates row permutations for a dense matrix, P * M' = M. 
  * Overwrites p from index 0 to min(ny, nx) - 1. P[y, x] = 1 is transformed to p[y] = x to conserve space.
  * The input matrix is overwritten with M', where the diagonal is made as large as possible.
  * There is no limitations on the input matrix size, except that ld, the horizontal offset, needs to be not less than nx.
  */

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