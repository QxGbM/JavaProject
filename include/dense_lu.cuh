#ifndef DENSE_LU
#define DENSE_LU

#include <stdio.h>
#include <cuda.h>

#include <helper_functions.h>
#include <cuda_helper_functions.cuh>

__global__ void dense_getrf_kernel (double *matrix, const unsigned nx, const unsigned ld, const unsigned ny)
{
  /* 
  * Using 1 block, running parallel both horizontal and vertical 
  * nx, ny cannot exceed BLOCK_SIZE, ld has no limitation 
  */
  const unsigned x = threadIdx.x, y = threadIdx.y;
  __shared__ double shm_matrix[BLOCK_SIZE * BLOCK_SIZE];
  const unsigned ld_aligned = ((ld + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

  if (x < nx && y < ny) { shm_matrix[y * BLOCK_SIZE + x] = matrix[y * ld_aligned + x]; }
  __syncthreads();

  for (unsigned i = 0; i < nx && i < ny; i++)
  {
    if (x == i && y > i && y < ny) { shm_matrix[y * BLOCK_SIZE + x] /= shm_matrix[i * BLOCK_SIZE + x]; }
    __syncthreads();
    if (x > i && x < nx && y > i && y < ny) { shm_matrix[y * BLOCK_SIZE + x] -= shm_matrix[y * BLOCK_SIZE + i] * shm_matrix[i * BLOCK_SIZE + x]; }
    __syncthreads();
  }
  
  if (x < nx && y < ny) { matrix[y * ld_aligned + x] = shm_matrix[y * BLOCK_SIZE + x]; }

}

__global__ void dense_trsm_left_kernel (double *X_B, double *A, const unsigned nx_x, const unsigned nx_a, const unsigned ld_x, const unsigned ld_a, 
  const unsigned ny_ab, const double alpha, const bool unit_triangular, const bool use_lower, const bool use_upper)
{
  /* 
  * Triangular solve for A.x = alpha * B, where A is already LU decomposed
  * Each block, solves A.x_i = B_i, and overwrites B_i using x_i
  * B_i is found using B_0 + i * nx_x, make sure ld_x is large enough so it does not circulate
  * nx_x, nx_a, ny_ab cannot exceed BLOCK_SIZE, ld_x, ld_a has no limitation, A's ny needs to match B's ny
  *
  * if nx_a < ny_ab, because the solution x has dim nx_a * nx_x, rows between nx_a and ny_ab are written with 0
  * otherwise if nx_a > ny_ab, nx_a is trimmed to ny_ab, since these column entries are redundant for solving x
  *
  * NOTE: this kernel assumes A.x = B is solvable and always generates a solution, even though it may not always satisfy
  * Ex. A is zero matrix and B is not
  */
  const unsigned x = threadIdx.x, y = threadIdx.y, b = blockIdx.x;
  __shared__ double shm_x[BLOCK_SIZE * BLOCK_SIZE], shm_a[BLOCK_SIZE * BLOCK_SIZE];
  const unsigned ld_x_aligned = ((ld_x + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
  const unsigned ld_a_aligned = ((ld_a + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

  if (b > 0) { X_B = &X_B[b * nx_x]; }
  if (x < nx_x && y < ny_ab && alpha != 0) { shm_x[y * BLOCK_SIZE + x] = alpha * X_B[y * ld_x_aligned + x]; }
  if (x < nx_a && y < ny_ab) { shm_a[y * BLOCK_SIZE + x] = A[y * ld_a_aligned + x]; }
  __syncthreads();

  for (unsigned i = 0; i < ny_ab; i++) 
  {
    if (use_lower && i < y && y < ny_ab && x < nx_x) { shm_x[y * BLOCK_SIZE + x] -= shm_a[y * BLOCK_SIZE + i] * shm_x[i * BLOCK_SIZE + x]; }
    __syncthreads();
  }

  for (unsigned i = 0; i < ny_ab; i++)
  {
    unsigned row = ny_ab - i - 1;
    if(!unit_triangular && y == 0 && x < nx_x) { shm_x[row * BLOCK_SIZE + x] /= shm_a[row * BLOCK_SIZE + row]; }
    __syncthreads();
    if(use_upper && y < row && x < nx_x) { shm_x[y * BLOCK_SIZE + x] -= shm_a[y * BLOCK_SIZE + row] * shm_x[row * BLOCK_SIZE + x]; }
    __syncthreads();
  }

  if (x < nx_x && y < ny_ab) { X_B[y * ld_x_aligned + x] = shm_x[y * BLOCK_SIZE + x]; }

}

__global__ void dense_trsm_right_kernel (double *X_B, double *A, const unsigned nx_ab, const unsigned ld_x, const unsigned ld_a, const unsigned ny_x, 
  const unsigned ny_a, const double alpha, const bool unit_triangular, const bool use_lower, const bool use_upper)
{
  /* Triangular solve for x.A = alpha * B, where A is already LU decomposed
  * Each block, solves x_i.A = B_i, and overwrites B_i using x_i
  * B_i is found using B_0 + i * ny_x * ld_x, make sure ny_x is not too large so it does not access any outside memory fragments
  * nx_ab, ny_x, ny_a cannot exceed BLOCK_SIZE, ld_x, ld_a has no limitation, A's nx needs to match B's nx
  *
  * if ny_a < nx_ab, because the solution x has dim ny_x * ny_a, columns between ny_a and nx_ab are written with 0
  * otherwise if ny_a > nx_ab, ny_a is trimmed to nx_ab, since these row entries are redundant for solving x
  *
  * NOTE: this kernel assumes x.A = B is solvable and always generates a solution, even though it may not always satisfy
  * Ex. A is zero matrix and B is not
  */
  const unsigned x = threadIdx.x, y = threadIdx.y, b = blockIdx.x;
  __shared__ double shm_x[BLOCK_SIZE * BLOCK_SIZE], shm_a[BLOCK_SIZE * BLOCK_SIZE];
  const unsigned ld_x_aligned = ((ld_x + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
  const unsigned ld_a_aligned = ((ld_a + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

  if (b > 0) { X_B = &X_B[b * ny_x * ld_x_aligned]; }
  if (x < nx_ab && y < ny_x && alpha != 0) { shm_x[y * BLOCK_SIZE + x] = alpha * X_B[y * ld_x_aligned + x]; }
  if (x < nx_ab && y < ny_a) { shm_a[y * BLOCK_SIZE + x] = A[y * ld_a_aligned + x]; }
  __syncthreads();

  for (unsigned i = 0; i < nx_ab; i++) 
  {
    if (!unit_triangular && x == 0 && y < ny_x) { shm_x[y * BLOCK_SIZE + i] /= shm_a[i * BLOCK_SIZE + i]; }
    __syncthreads();
    if (use_upper && y < ny_x && i < x && x < nx_ab) { shm_x[y * BLOCK_SIZE + x] -= shm_x[y * BLOCK_SIZE + i] * shm_a[i * BLOCK_SIZE + x]; }
    __syncthreads();
  }

  for (unsigned i = 0; i < nx_ab; i++)
  {
    unsigned col = nx_ab - i - 1;
    if (use_lower && y < ny_x && x < col) { shm_x[y * BLOCK_SIZE + x] -= shm_x[y * BLOCK_SIZE + col] * shm_a[col * BLOCK_SIZE + x]; }
    __syncthreads();
  }

  if (x < nx_ab && y < ny_x) { X_B[y * ld_x_aligned + x] = shm_x[y * BLOCK_SIZE + x]; }

}

__global__ void dense_gemm_kernel (double *X_C, double *A, double *B, const unsigned nx_bx, const unsigned nx_a_ny_b, const unsigned ny_ax, 
  const unsigned ld_x, const unsigned ld_a, const unsigned ld_b, const double alpha, const double beta, const unsigned b2)
{
  /* General Matrix-Matrix multiplication, for x = alpha * A * B + beta * C
  * Each block, does x_ij = alpha * A_ik * B_kj + beta * C_ij, and overwrites C_ij using x_ij
  * A_ik is found using A_00 + i * ny_ax * ld_a + k * nx_a_ny_b
  * B_kj is found using B_00 + k * nx_a_ny_b * ld_b + j * nx_bx
  * C_ij is found using C_00 + i * ny_ax * ld_x + j * nx_bx
  *
  * A's ny needs to match C's ny, A's nx needs to match B's ny, and B's nx needs to match C's nx
  * nx_bx, nx_a_ny_b, ny_ax cannot exceed BLOCK_SIZE, ld_x, ld_a, ld_b has no limitation
  * 
  * k is being passed as the last parameter, to prevent simultaneous writing to one block
  */
  const unsigned x = threadIdx.x, y = threadIdx.y, b0 = blockIdx.x, b1 = blockIdx.y;
  __shared__ double shm_x[BLOCK_SIZE * BLOCK_SIZE], shm_a[BLOCK_SIZE * BLOCK_SIZE], shm_b[BLOCK_SIZE * BLOCK_SIZE];
  const unsigned ld_x_aligned = ((ld_x + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
  const unsigned ld_a_aligned = ((ld_a + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
  const unsigned ld_b_aligned = ((ld_b + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

  if (b0 > 0) { X_C = &X_C[b0 * ny_ax * ld_x_aligned]; A = &A[b0 * ny_ax * ld_a_aligned]; }
  if (b1 > 0) { X_C = &X_C[b1 * nx_bx]; B = &B[b1 * nx_bx]; }
  if (x < nx_bx && y < ny_ax && beta != 0) { shm_x[y * BLOCK_SIZE + x] = beta * X_C[y * ld_x_aligned + x]; }

  for (unsigned i = 0; i < b2; i++) /* iteration for k */
  {
    if (i > 0) { A = &A[nx_a_ny_b]; B = &B[nx_a_ny_b * ld_b_aligned]; }
    if (x < nx_a_ny_b && y < ny_ax) { shm_a[y * BLOCK_SIZE + x] = A[y * ld_a_aligned + x]; }
    if (x < nx_bx && y < nx_a_ny_b) { shm_b[y * BLOCK_SIZE + x] = B[y * ld_b_aligned + x]; }
    __syncthreads();
    for (unsigned j = 0; j < nx_a_ny_b; j++) 
    {
      if (x < nx_bx && y < ny_ax) { shm_x[y * BLOCK_SIZE + x] += alpha * shm_a[y * BLOCK_SIZE + j] * shm_b[j * BLOCK_SIZE + x]; }
    }
    __syncthreads();
  }

  if (x < nx_bx && y < ny_ax) { X_C[y * ld_x_aligned + x] = shm_x[y * BLOCK_SIZE + x]; }

}

__host__ int dense_getrf_sync (double *matrix, const unsigned nx, const unsigned ld, const unsigned ny) 
{
  if (ld < nx) { printf("GETRF ABORT: Matrix's horizontal offset is less than the number of entries.\n");  return -1; }
  double *dev_matrix = 0;

  cudaStream_t main_stream;
  cudaStreamCreate(&main_stream);

  matrix_copy_toDevice_sync (matrix, &dev_matrix, nx, ld, ny);
  create_timing_event_to_stream ("GETRF TOTAL", main_stream);

  if (nx <= BLOCK_SIZE && ny <= BLOCK_SIZE) 
  {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE), grid(1);
    dense_getrf_kernel <<<grid, block, 0, main_stream>>> (dev_matrix, nx, ld, ny);
  }
  else 
  {
    const unsigned ld_aligned = ((ld + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE; // use aligned ld for dev_matrix.
    unsigned min_n = (nx > ny) ? ny : nx;
    unsigned diag_n = (min_n > BLOCK_SIZE) ? BLOCK_SIZE : min_n; // diagnal block dim.
    unsigned new_nx = nx - diag_n; // top right block x dim.
    unsigned new_ny = ny - diag_n; // bottom left block y dim.
    double *dev_matrix_c = dev_matrix, *dev_matrix_a = dev_matrix, *dev_matrix_b = dev_matrix; 
    // current top-left block pointer, bottom left block pointer, top right block pointer.

    cudaStream_t sup_stream;
    cudaStreamCreate(&sup_stream);

    bool looping = true;
    do
    {
      dim3 block(BLOCK_SIZE, BLOCK_SIZE), grid_1x1(1);
      create_timing_event_to_stream ("GETRF", main_stream);
      dense_getrf_kernel <<<grid_1x1, block, 0, main_stream>>> (dev_matrix_c, diag_n, ld, diag_n);
      create_timing_event_to_stream ("GETRF", main_stream);
      // both aligned ld & ld are acceptable. ld alignment is done in kernels too.
      cudaStreamSynchronize(main_stream);

      const unsigned ny_a = (new_ny > BLOCK_SIZE) ? BLOCK_SIZE : new_ny; // # entries inside a block, cannot exceed block size.
      const unsigned nx_b = (new_nx > BLOCK_SIZE) ? BLOCK_SIZE : new_nx; // # entries inside a block, cannot exceed block size.

      if (new_ny > 0)
      {
        dev_matrix_a = &dev_matrix_c[diag_n * ld_aligned];
        dim3 grid_1xn((new_ny + BLOCK_SIZE - 1) / BLOCK_SIZE); // (ny / Block size)'s ceiling.
        
        create_timing_event_to_stream ("TRSM RIGHT", sup_stream);
        dense_trsm_right_kernel <<<grid_1xn, block, 0, sup_stream>>> 
        (dev_matrix_a, dev_matrix_c, diag_n, ld, ld, ny_a, diag_n, 1, false, false, true);
        create_timing_event_to_stream ("TRSM RIGHT", sup_stream);
      }

      if (new_nx > 0)
      {
        dev_matrix_b = &dev_matrix_c[diag_n];
        dim3 grid_1xn((new_nx + BLOCK_SIZE - 1) / BLOCK_SIZE); // (nx / Block size)'s ceiling.
        
        create_timing_event_to_stream ("TRSM LEFT", main_stream);
        dense_trsm_left_kernel <<<grid_1xn, block, 0, main_stream>>> 
        (dev_matrix_b, dev_matrix_c, nx_b, diag_n, ld, ld, diag_n, 1, true, true, false);
        create_timing_event_to_stream ("TRSM LEFT", main_stream);
      }

      if (new_nx > 0 && new_ny > 0)
      { 
        dev_matrix_c = &dev_matrix_c[diag_n * ld_aligned + diag_n];
        dim3 grid_nxn((new_ny + BLOCK_SIZE - 1) / BLOCK_SIZE, (new_nx + BLOCK_SIZE - 1) / BLOCK_SIZE);
  
        cudaStreamSynchronize(sup_stream);
        cudaStreamSynchronize(main_stream);
        create_timing_event_to_stream ("GEMM", main_stream);
        dense_gemm_kernel <<<grid_nxn, block, 0, main_stream>>> 
        (dev_matrix_c, dev_matrix_a, dev_matrix_b, nx_b, diag_n, ny_a, ld, ld, ld, -1, 1, 1);
        create_timing_event_to_stream ("GEMM", main_stream);
  
        min_n = (new_nx > new_ny) ? new_ny : new_nx;
        diag_n = (min_n > BLOCK_SIZE) ? BLOCK_SIZE : min_n;
        new_nx -= diag_n;
        new_ny -= diag_n;
      }
      else { looping = false; }

    } while (looping);
    cudaStreamDestroy(sup_stream);

  }
  create_timing_event_to_stream ("GETRF TOTAL", main_stream);
  cudaStreamDestroy(main_stream);

  device_sync_dump_timed_events ();
  printf("Cuda Execution: getrf finished.\n");

  matrix_copy_toHost_sync (&dev_matrix, matrix, nx, ld, ny, true);
  cudaDeviceReset();
  return 0;
}

__host__ int dense_trsm_sync (double *matrix_x, const unsigned nx_x, const unsigned ld_x, const unsigned ny_x,
  double *matrix_a, const unsigned nx_a, const unsigned ld_a, const unsigned ny_a, const double alpha, const bool side, 
  const bool unit_triangular, const bool use_lower, const bool use_upper)
{
  if ((!side && ny_a != ny_x) || (side && nx_a != nx_x)) { printf("TRSM ABORT: Cannot match matrix A and B's dimensions.\n");  return -1; }
  if (ld_x < nx_x) { printf("TRSM ABORT: Matrix x's horizontal offset is less than the number of entries.\n");  return -1; }
  if (ld_a < nx_a) { printf("TRSM ABORT: Matrix a's horizontal offset is less than the number of entries.\n");  return -1; }

  if (unit_triangular && !use_lower && !use_upper) { printf("TRSM WARNING: A is being configured as identity Matrix.\n"); }

  double *dev_matrix_x = 0, *dev_matrix_a = 0;

  cudaStream_t main_stream;
  cudaStreamCreate(&main_stream);

  matrix_copy_toDevice_sync (matrix_x, &dev_matrix_x, nx_x, ld_x, ny_x);
  matrix_copy_toDevice_sync (matrix_a, &dev_matrix_a, nx_a, ld_a, ny_a);

  if (ny_a <= BLOCK_SIZE && nx_a <= BLOCK_SIZE)
  {
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    if (!side) /* left */ 
    { 
      const unsigned blocks = (nx_x + BLOCK_SIZE - 1) / BLOCK_SIZE;
      const unsigned entries = (nx_x > BLOCK_SIZE) ? BLOCK_SIZE : nx_x;
      dim3 grid(blocks);
      dense_trsm_left_kernel <<<grid, block, 0, main_stream>>> 
      (dev_matrix_x, dev_matrix_a, entries, nx_a, ld_x, ld_a, ny_a, alpha, unit_triangular, use_lower, use_upper); 
    }
    else /* right */ 
    {
      const unsigned blocks = (ny_x + BLOCK_SIZE - 1) / BLOCK_SIZE;
      const unsigned entries = (ny_x > BLOCK_SIZE) ? BLOCK_SIZE : nx_x;
      dim3 grid(blocks);
      dense_trsm_right_kernel <<<grid, block, 0, main_stream>>> 
      (dev_matrix_x, dev_matrix_a, nx_a, ld_x, ld_a, entries, ny_a, alpha, unit_triangular, use_lower, use_upper); 
    }
  }
  else {
    const unsigned ld_a_aligned = ((ld_a + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    const unsigned ld_x_aligned = ((ld_x + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
    const unsigned n = (((nx_a > ny_a) ? ny_a : nx_a) + BLOCK_SIZE - 1) / BLOCK_SIZE; // # of iterations needed.

    if (!side) /* left */ 
    {
      for (unsigned i = 0; i < n; i++) /* solve for L */
      {
        const unsigned rc = i; // current row and column block number.
        const unsigned rc_real = rc * BLOCK_SIZE; // actual number of rows/cols processed.
        double *dev_matrix_a_diag = &dev_matrix_a[rc_real * ld_a_aligned + rc_real]; // current diagonal block ptr.

        const unsigned rows = ny_a - rc_real; // current size for a
        const unsigned cols = nx_a - rc_real; // current size for a
        const unsigned current_rows = (rows > BLOCK_SIZE) ? BLOCK_SIZE : rows; // diagonal block ny.
        const unsigned current_cols = (cols > BLOCK_SIZE) ? BLOCK_SIZE : cols; // diagonal block nx.

        double *dev_matrix_x_top = &dev_matrix_x[rc_real * ld_x_aligned]; // same row as diagonal block.

        const unsigned blocks_x = (nx_x + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const unsigned entries_x = (nx_x > BLOCK_SIZE) ? BLOCK_SIZE : nx_x;
        dim3 grid_1xn(blocks_x);
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);

        dense_trsm_left_kernel <<<grid_1xn, block, 0, main_stream>>> 
        (dev_matrix_x_top, dev_matrix_a_diag, entries_x, current_cols, ld_x, ld_a, current_rows, alpha, true, use_lower, false);

        const unsigned rows_gemm = rows - current_rows; // # of rows that needs gemm
        const unsigned cols_gemm = current_cols; // # of cols that needs gemm

        if(use_lower && rows_gemm > 0 && cols_gemm > 0)
        {
          double *dev_matrix_a_left = &dev_matrix_a[(rc_real + current_rows) * ld_a_aligned + rc_real]; // 1 block below from diagonal.
          double *dev_matrix_x_left = &dev_matrix_x[(rc_real + current_rows) * ld_x_aligned]; // next row from solved.

          const unsigned blocks_y = (rows_gemm + BLOCK_SIZE - 1) / BLOCK_SIZE;
          const unsigned entries_y = (rows_gemm > BLOCK_SIZE) ? BLOCK_SIZE : rows_gemm;
          dim3 grid_nxn(blocks_y, blocks_x);

          dense_gemm_kernel <<<grid_nxn, block, 0, main_stream>>> 
          (dev_matrix_x_left, dev_matrix_a_left, dev_matrix_x_top, entries_x, cols_gemm, entries_y, ld_x, ld_a, ld_x, -1, 1, 1);
        }
      }

      for (unsigned i = 0; i < n; i++) /* solve for U */
      {
        const unsigned rc = n - (i + 1); // current row and column block number.
        const unsigned rc_real = rc * BLOCK_SIZE; // actual number of rows/cols processed.
        double *dev_matrix_a_diag = &dev_matrix_a[rc_real * ld_a_aligned + rc_real]; // current diagonal block ptr.

        const unsigned rows = ny_a - rc_real; // current size for a
        const unsigned cols = nx_a - rc_real; // current size for a
        const unsigned current_rows = (rows > BLOCK_SIZE) ? BLOCK_SIZE : rows; // diagonal block ny.
        const unsigned current_cols = (cols > BLOCK_SIZE) ? BLOCK_SIZE : cols; // diagonal block nx.

        double *dev_matrix_x_top = &dev_matrix_x[rc_real * ld_x_aligned]; // same row as diagonal block.

        const unsigned blocks_x = (nx_x + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const unsigned entries_x = (nx_x > BLOCK_SIZE) ? BLOCK_SIZE : nx_x;
        dim3 grid_1xn(blocks_x);
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);

        dense_trsm_left_kernel <<<grid_1xn, block, 0, main_stream>>> 
        (dev_matrix_x_top, dev_matrix_a_diag, entries_x, current_cols, ld_x, ld_a, current_rows, alpha, unit_triangular, false, use_upper);

        const unsigned rows_gemm = ny_a - rows; // # of rows that needs gemm
        const unsigned cols_gemm = current_cols; // # of cols that needs gemm

        if(use_upper && rows_gemm > 0 && cols_gemm > 0)
        {
          double *dev_matrix_a_top = &dev_matrix_a[rc_real]; // the top row block from diagonal.
          double *dev_matrix_x_left = &dev_matrix_x[0]; 

          const unsigned blocks_y = (rows_gemm + BLOCK_SIZE - 1) / BLOCK_SIZE;
          const unsigned entries_y = (rows_gemm > BLOCK_SIZE) ? BLOCK_SIZE : rows_gemm;
          dim3 grid_nxn(blocks_y, blocks_x);
  
          dense_gemm_kernel <<<grid_nxn, block, 0, main_stream>>> 
          (dev_matrix_x_left, dev_matrix_a_top, dev_matrix_x_top, entries_x, cols_gemm, entries_y, ld_x, ld_a, ld_x, -1, 1, 1);
        }
      }
    }
    else /* right */ 
    {
      for (unsigned i = 0; i < n; i++) /* solve for U */
      {
        const unsigned rc = i; // current row and column block number.
        const unsigned rc_real = rc * BLOCK_SIZE; // actual number of rows/cols processed.
        double *dev_matrix_a_diag = &dev_matrix_a[rc_real * ld_a_aligned + rc_real]; // current diagonal block ptr.

        const unsigned rows = ny_a - rc_real; // current size for a
        const unsigned cols = nx_a - rc_real; // current size for a
        const unsigned current_rows = (rows > BLOCK_SIZE) ? BLOCK_SIZE : rows; // diagonal block ny.
        const unsigned current_cols = (cols > BLOCK_SIZE) ? BLOCK_SIZE : cols; // diagonal block nx.

        double *dev_matrix_x_left = &dev_matrix_x[rc_real]; // same column as diagonal block.

        const unsigned blocks_y = (ny_x + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const unsigned entries_y = (ny_x > BLOCK_SIZE) ? BLOCK_SIZE : ny_x;
        dim3 grid_1xn(blocks_y);
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);

        dense_trsm_right_kernel <<<grid_1xn, block, 0, main_stream>>> 
        (dev_matrix_x_left, dev_matrix_a_diag, current_cols, ld_x, ld_a, entries_y, current_rows, alpha, unit_triangular, false, use_upper);

        const unsigned rows_gemm = current_rows; // # of rows that needs gemm
        const unsigned cols_gemm = cols - current_cols; // # of cols that needs gemm

        if(use_upper && rows_gemm > 0 && cols_gemm > 0)
        {
          double *dev_matrix_a_top = &dev_matrix_a[rc_real * ld_a_aligned + rc_real + current_cols]; // the 1 block right from diagonal.
          double *dev_matrix_x_top = &dev_matrix_x[rc_real + current_cols]; // next column from solved.

          const unsigned blocks_x = (cols_gemm + BLOCK_SIZE - 1) / BLOCK_SIZE;
          const unsigned entries_x = (cols_gemm > BLOCK_SIZE) ? BLOCK_SIZE : cols_gemm;
          dim3 grid_nxn(blocks_y, blocks_x);
  
          dense_gemm_kernel <<<grid_nxn, block, 0, main_stream>>> 
          (dev_matrix_x_top, dev_matrix_x_left, dev_matrix_a_top, entries_x, rows_gemm, entries_y, ld_x, ld_x, ld_a, -1, 1, 1);
        }
      }

      for (unsigned i = 0; i < n; i++) /* solve for L */
      {
        const unsigned rc = n - (i + 1); // current row and column block number.
        const unsigned rc_real = rc * BLOCK_SIZE; // actual number of rows/cols processed.
        double *dev_matrix_a_diag = &dev_matrix_a[rc_real * ld_a_aligned + rc_real]; // current diagonal block ptr.

        const unsigned rows = ny_a - rc_real; // current size for a
        const unsigned cols = nx_a - rc_real; // current size for a
        const unsigned current_rows = (rows > BLOCK_SIZE) ? BLOCK_SIZE : rows; // diagonal block ny.
        const unsigned current_cols = (cols > BLOCK_SIZE) ? BLOCK_SIZE : cols; // diagonal block nx.

        double *dev_matrix_x_right = &dev_matrix_x[rc_real]; // same column as diagonal block.

        const unsigned blocks_y = (ny_x + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const unsigned entries_y = (ny_x > BLOCK_SIZE) ? BLOCK_SIZE : ny_x;
        dim3 grid_1xn(blocks_y);
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);

        dense_trsm_right_kernel <<<grid_1xn, block, 0, main_stream>>> 
        (dev_matrix_x_right, dev_matrix_a_diag, current_cols, ld_x, ld_a, entries_y, current_rows, alpha, true, use_lower, false);

        const unsigned rows_gemm = current_rows; // # of rows that needs gemm
        const unsigned cols_gemm = nx_a - cols; // # of cols that needs gemm

        if(use_lower && rows_gemm > 0 && cols_gemm > 0)
        {
          double *dev_matrix_a_bot = &dev_matrix_a[rc_real * ld_a_aligned]; // left most block from diagonal.
          double *dev_matrix_x_left = &dev_matrix_x[0];

          const unsigned blocks_x = (cols_gemm + BLOCK_SIZE - 1) / BLOCK_SIZE;
          const unsigned entries_x = (cols_gemm > BLOCK_SIZE) ? BLOCK_SIZE : cols_gemm;
          dim3 grid_nxn(blocks_y, blocks_x);

          dense_gemm_kernel <<<grid_nxn, block, 0, main_stream>>> 
          (dev_matrix_x_left, dev_matrix_x_right, dev_matrix_a_bot, entries_x, rows_gemm, entries_y, ld_x, ld_x, ld_a, -1, 1, 1);
        }
      }
    }

  }
  cudaStreamDestroy(main_stream);

  cudaDeviceSynchronize();
  printf("Cuda Execution: trsm finished.\n");

  matrix_copy_toHost_sync (&dev_matrix_x, matrix_x, nx_x, ld_x, ny_x, true);
  cudaFree(dev_matrix_a);
  printf("Freed matrix a in cuda global memory.\n");

  cudaDeviceReset();
  return 0;

}

__host__ int dense_gemm_sync (double *matrix_x, const unsigned nx_x, const unsigned ld_x, const unsigned ny_x,
  double *matrix_a, const unsigned nx_a, const unsigned ld_a, const unsigned ny_a, 
  double *matrix_b, const unsigned nx_b, const unsigned ld_b, const unsigned ny_b, const double alpha, const double beta)
{
  if ((nx_a != ny_b) || (ny_a != ny_x) || (nx_b != nx_x)) { printf("GEMM ABORT: Cannot match matrices' dimensions.\n");  return -1; }
  if (ld_x < nx_x) { printf("GEMM ABORT: Matrix x's horizontal offset is less than the number of entries.\n");  return -1; }
  if (ld_a < nx_a) { printf("GEMM ABORT: Matrix a's horizontal offset is less than the number of entries.\n");  return -1; }
  if (ld_b < nx_b) { printf("GEMM ABORT: Matrix b's horizontal offset is less than the number of entries.\n");  return -1; }

  double *dev_matrix_x = 0, *dev_matrix_a = 0, *dev_matrix_b = 0;

  cudaStream_t main_stream;
  cudaStreamCreate(&main_stream);

  matrix_copy_toDevice_sync (matrix_x, &dev_matrix_x, nx_x, ld_x, ny_x);
  matrix_copy_toDevice_sync (matrix_a, &dev_matrix_a, nx_a, ld_a, ny_a);
  matrix_copy_toDevice_sync (matrix_b, &dev_matrix_b, nx_b, ld_b, ny_b);

  const unsigned block_i = (ny_x + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const unsigned block_j = (nx_x + BLOCK_SIZE - 1) / BLOCK_SIZE;
  const unsigned block_k = (nx_a + BLOCK_SIZE - 1) / BLOCK_SIZE;

  const unsigned entries_i = (ny_x > BLOCK_SIZE) ? BLOCK_SIZE : ny_x;
  const unsigned entries_j = (nx_x > BLOCK_SIZE) ? BLOCK_SIZE : nx_x;
  const unsigned entries_k = (nx_a > BLOCK_SIZE) ? BLOCK_SIZE : nx_a;

  dim3 block(BLOCK_SIZE, BLOCK_SIZE), grid(block_i, block_j);
  dense_gemm_kernel <<<grid, block, 0, main_stream>>> 
  (dev_matrix_x, dev_matrix_a, dev_matrix_b, entries_j, entries_k, entries_i, ld_x, ld_a, ld_b, alpha, beta, block_k);

  cudaStreamDestroy(main_stream);

  cudaDeviceSynchronize();
  printf("Cuda Execution: gemm finished.\n");

  matrix_copy_toHost_sync (&dev_matrix_x, matrix_x, nx_x, ld_x, ny_x, true);
  cudaFree(dev_matrix_a);
  printf("Freed matrix a in cuda global memory.\n");
  cudaFree(dev_matrix_b);
  printf("Freed matrix b in cuda global memory.\n");

  cudaDeviceReset();
  return 0;

}

__host__ int dense_getrf_sync (Matrix *m)
{
  double *matrix = m -> head;
  const unsigned nx = m -> nx, ld = m -> ld, ny = m -> ny;

  return dense_getrf_sync (matrix, nx, ld, ny);
}

__host__ int dense_trsm_sync (Matrix *x_B, Matrix *A, const double alpha, const bool side, 
  const bool unit_triangular, const bool use_lower, const bool use_upper)
{
  double *matrix_x = x_B -> head, *matrix_a = A -> head;
  const unsigned nx_x = x_B -> nx, ld_x = x_B -> ld, ny_x = x_B -> ny;
  const unsigned nx_a = A -> nx, ld_a = A -> ld, ny_a = A -> ny;

  return dense_trsm_sync (matrix_x, nx_x, ld_x, ny_x, matrix_a, nx_a, ld_a, ny_a, alpha, side, unit_triangular, use_lower, use_upper);
}

__host__ int dense_gemm_sync (Matrix *x_C, Matrix *A, Matrix *B, const double alpha, const double beta)
{
  double *matrix_x = x_C -> head, *matrix_a = A -> head, *matrix_b = B -> head;
  const unsigned nx_x = x_C -> nx, ld_x = x_C -> ld, ny_x = x_C -> ny;
  const unsigned nx_a = A -> nx, ld_a = A -> ld, ny_a = A -> ny;
  const unsigned nx_b = B -> nx, ld_b = B -> ld, ny_b = B -> ny;

  return dense_gemm_sync (matrix_x, nx_x, ld_x, ny_x, matrix_a, nx_a, ld_a, ny_a, matrix_b, nx_b, ld_b, ny_b, alpha, beta);
}

#endif