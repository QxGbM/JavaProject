#ifndef _SVD_CUH
#define _SVD_CUH

#include <pspl.cuh>

template <class T> __device__ void blockSwapColumns (T * col1, T * col2, const int ny, const int ld)
{
  for (int i = thread_rank(); i < ny; i += block_dim())
  {
    const T t = col1[i * ld]; 
    col1[i * ld] = col2[i * ld]; 
    col2[i * ld] = t;
  }
}

template <class T> __device__ void blockRotateColumns (T * col1, T * col2, const int ny, const int ld, const double sine, const double cosine)
{
  for (int i = thread_rank(); i < ny; i += block_dim())
  {
    const T col1_T = col1[i * ld], col2_T = col2[i * ld];
    col1[i * ld] = cosine * col1_T - sine * col2_T;
    col2[i * ld] = sine * col1_T + cosine * col2_T;
  }
}

template <class T> __device__ T blockVectorMultiplication (const T * vec1, const T * vec2, const int length, const int ld_1, const int ld_2)
{
  T thread_sum = 0;
  for (int i = thread_rank(); i < length; i += block_dim())
  { thread_sum += vec1[i * ld_1] * vec2[i * ld_2]; }

  for (int mask = warpSize / 2; mask > 0; mask /= 2)
  { thread_sum += __shfl_xor_sync (0xffffffff, thread_sum, mask, warpSize); }

  __shared__ T warp_sum[MAX_WARPS];
  if (lane_rank() == 0)
  { warp_sum[warp_rank()] = thread_sum; }
  __syncthreads();

  if (block_dim() > warpSize && warp_rank() == 0)
  {
    thread_sum = 0;
    for (int i = lane_rank(); i < num_warps(); i += warpSize)
    { thread_sum += warp_sum[i]; }
    
    for (int mask = warpSize / 2; mask > 0; mask /= 2)
    { thread_sum += __shfl_xor_sync (0xffffffff, thread_sum, mask, warpSize); }
    if (lane_rank() == 0) 
    { warp_sum[0] = thread_sum; }
  }
  __syncthreads();

  return warp_sum[0];
}

template <class T> __device__ double blockJacobiRotate (T * A, T * VT, const int nx, const int ny, const int ld_a, const int ld_v, const int col_i, const int col_j)
{
  const T s_ii = blockVectorMultiplication <T> (&A[col_i], &A[col_i], ny, ld_a, ld_a);
  const T s_jj = blockVectorMultiplication <T> (&A[col_j], &A[col_j], ny, ld_a, ld_a);
  const T s_ij = blockVectorMultiplication <T> (&A[col_i], &A[col_j], ny, ld_a, ld_a);

  if (s_ii > s_jj)
  {
    blockSwapColumns <T> (&A[col_i], &A[col_j], ny, ld_a);
    blockSwapColumns <T> (&VT[col_i], &VT[col_j], nx, ld_v);
  }

  __shared__ double sine, cosine, conv;
  if (thread_rank() == 0)
  {
    const double torque = (s_jj - s_ii) / (2.0 * s_ij);
    const double sign_torque = (torque >= 0) ? 1.0 : -1.0;
    const double tangent = sign_torque / (abs(torque) + sqrt(1.0 + torque * torque));
    cosine = 1.0 / (sqrt(1.0 + tangent * tangent));
    sine = cosine * tangent;
    conv = (s_ij > 0) ? (double) s_ij : (double) -s_ij;
  }
  __syncthreads();

  blockRotateColumns <T> (&A[col_i], &A[col_j], ny, ld_a, sine, cosine);
  blockRotateColumns <T> (&VT[col_i], &VT[col_j], nx, ld_v, sine, cosine);
  __syncthreads();

  return conv;
}

template <class T> __device__ int blockJacobiSVD (T * A, T * VT, const int nx, const int ny, const int ld_a, const int ld_v, const double epi, const int iter_limit)
{
  __shared__ bool iter;
  __shared__ int iter_counter;
  if (thread_rank() == 0) 
  { iter_counter = 0; }

  do
  {
    if (thread_rank() == 0)
    { iter = false; iter_counter ++; }

    for (int i = 1; i < nx; i++)
    {
      for (int j = 0; j < i; j++)
      {
        double conv = blockJacobiRotate <T> (A, VT, nx, ny, ld_a, ld_v, i, j);
        if (thread_rank() == 0 && conv > epi)
        { iter = true; }
      }
    }
  } while (iter && iter_counter < iter_limit);
  return iter_counter;
}

__global__ void svd_kernel(double * A, double * VT, const int nx, const int ny, const int ld_a, const int ld_v)
{
  int i = blockJacobiSVD <double> (A, VT, nx, ny, ld_a, ld_v, 1.0e-14, 100);
  if (thread_rank() == 0) { printf("iters: %d\n", i); }
}

void swap_col(double *col1, double *col2, const int ny, const int ld)
{
  for (int i = 0; i < ny; i++)
  {
    const double t = col1[i * ld]; col1[i * ld] = col2[i * ld]; col2[i * ld] = t;
  }
}


int test1 () 
{

  const int nx = 16, ny = 16;
  
  dev_dense <double> *d_VT, *d_A;

  d_A = new dev_dense <double> (nx, ny);
  d_A -> loadTestMatrix(20);

  d_VT = new dev_dense <double> (nx, nx);
  d_VT -> loadIdentityMatrix();

  double *A = d_A -> getElements();
  double *VT = d_VT -> getElements();

  timer myTimer = timer();

  myTimer.newEvent("GETRF", start);
  svd_kernel <<<1, 1024>>> (A, VT, nx, ny, nx, nx);
  myTimer.newEvent("GETRF", end);

  myTimer.dumpAllEvents_Sync();

/*svd:
  bool iter = false;

  for(int i = 1; i < nx; i++)
  {
    for(int j = 0; j < i; j++)
    {

      double s_ii = 0.0, s_jj = 0.0, s_ij = 0.0;

      for(int k = 0; k < ny; k++)
      {
        s_ii += A[k * nx + i] * A[k * nx + i];
        s_jj += A[k * nx + j] * A[k * nx + j];
        s_ij += A[k * nx + i] * A[k * nx + j];
      }

      if (s_ii > s_jj) 
      { 
        swap_col(&A[i], &A[j], ny, nx); 
        swap_col(&VT[i], &VT[j], nx, nx);
      }

      const double torque = (s_jj - s_ii) / (2.0 * s_ij);
      const double sign_torque = (torque >= 0) ? 1.0 : -1.0;
      const double t = sign_torque / (abs (torque) + sqrt (1.0 + torque * torque));
      const double c = 1.0 / (sqrt (1.0 + t * t));
      const double s = c * t;

      for (int k = 0; k < ny; k++)
      {
        const double ai_T = A[k * nx + i], aj_T = A[k * nx + j];
        A[k * nx + i] = c * ai_T - s * aj_T;
        A[k * nx + j] = s * ai_T + c * aj_T;
      }

      for (int k = 0; k < nx; k++)
      {
        const double vi_T = VT[k * nx + i], vj_T = VT[k * nx + j];
        VT[k * nx + i] = c * vi_T - s * vj_T;
        VT[k * nx + j] = s * vi_T + c * vj_T;
      }

      if (abs(s_ij) > 1.e-14) iter = true;

    }
  }

  if (iter) goto svd;*/

  for (int i = 0; i < nx; i++)
  {
    double s = 0.0;
    for (int j = 0; j < ny; j++)
    { s += A[j * nx + i] * A[j * nx + i]; }

    s = sqrt(s);
    printf("%d: %e\n", i, s);
  }

  dev_dense <double> *c = d_A -> matrixMultiplication(d_VT -> transpose());
  d_A -> loadTestMatrix(20);
  printf("Rel. L2 Error: %e\n\n", c -> L2Error(d_A));

  return 0;
}

#endif