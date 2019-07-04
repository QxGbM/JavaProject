#ifndef _DEV_LOW_RANK_FUNCS_CUH
#define _DEV_LOW_RANK_FUNCS_CUH

#include <pspl.cuh>


template <class T> __device__ T warpAllReduceSum (T value)
{
  for (int mask = warpSize / 2; mask > 0; mask /= 2)
  { value += __shfl_xor_sync(0xffffffff, value, mask, warpSize); }
  return value;
}

template <class T> __device__ T blockAllReduceSum (T value, T * shm)
{
  value = warpAllReduceSum <T> (value);

  if (lane_rank() == 0)
  { shm[warp_rank()] = value; }
  __syncthreads();

  if (block_dim() > warpSize && warp_rank() == 0)
  {
    value = 0;
    for (int i = lane_rank(); i < num_warps(); i += warpSize)
    { value += shm[i]; }

    value = warpAllReduceSum <T> (value);
    if (lane_rank() == 0)
    { shm[0] = value; }
  }
  __syncthreads();

  return shm[0];
}


template <class T> 
__device__ bool blockSingleSideJacobiSVD (T * __restrict__ UxS, T * __restrict__ VT, const int nx, const int ny, const int ld_UxS, const int ld_VT, T * __restrict__ shm, const T epi)
{
  bool iter = false;

  for (int step = nx / 2; step > 0; step /= 2) for (int offset = 0; offset < step; offset++)
  {
    for (int i = thread_rank(); i < 2 * nx; i += block_dim())
    { shm[i] = 0.; }
    __syncthreads();

    for (int row = 0; row < ny; row++)
    {
      const int row_UxS = row * ld_UxS;
      for (int col = thread_rank(); col < nx; col += block_dim())
      {
        const T e = UxS[row_UxS + col];
        shm[col] += e * e;
        const int lane = col % (2 * step);
        if (lane < step)
        { 
          if (lane + offset >= step)
          { shm[nx + col] += e * UxS[row_UxS + col + offset]; }
          else
          { shm[nx + col] += e * UxS[row_UxS + col + step + offset]; }
        }
      }
    }
    __syncthreads();

    for (int col = warp_rank(); col < nx; col += num_warps())
    {
      if (shm[nx + col] > epi || shm[nx + col] < -epi)
      {
        double sine, cosine;
        const int col2 = col + offset + ((col % step + offset >= step) ? 0 : step);
        const bool swap = shm[col] < shm[col2];
        iter = true;

        if (lane_rank() == 0)
        {
          const double torque = (shm[col2] - shm[col]) / (shm[nx + col] * 2.);
          const double tangent = (signbit(torque) * -2 + 1) / (fabs(torque) + sqrt(1. + torque * torque));
          cosine = rsqrt(1. + tangent * tangent);
          sine = cosine * tangent;
        }
        __syncwarp();

        cosine = __shfl_sync(0xffffffff, cosine, 0, warpSize);
        sine = __shfl_sync(0xffffffff, sine, 0, warpSize);

        for (int row = lane_rank(); row < ny; row += warpSize)
        {
          const int row_UxS = row * ld_UxS;
          const T e1 = swap ? UxS[row_UxS + col2] : UxS[row_UxS + col], e2 = swap ? UxS[row_UxS + col] : UxS[row_UxS + col2];
          UxS[row_UxS + col] = cosine * e1 - sine * e2;
          UxS[row_UxS + col2] = sine * e1 + cosine * e2;
        }

        for (int row = lane_rank(); row < nx; row += warpSize)
        {
          const int row_VT = row * ld_VT;
          const T e3 = swap ? VT[row_VT + col2] : VT[row_VT + col], e4 = swap ? VT[row_VT + col] : VT[row_VT + col2];
          VT[row_VT + col] = cosine * e3 - sine * e4;
          VT[row_VT + col2] = sine * e3 + cosine * e4;
        }
        __syncwarp();

      }
    }
    __syncthreads();
  }

  return iter;
}


template <class T> 
__device__ void blockGivensRotation (T * __restrict__ M, const int nx, const int ny, const int ld_m)
{
  const int l_id = lane_rank(), w_id = warp_rank(), tb_w_size = num_warps(), n = nx + ny - 2;

  for (int iter = 0; iter < n; iter++)
  {
    for (int col = w_id; col < nx; col += tb_w_size)
    {
      const int row = ny - 2 + 2 * col - iter;
      if (row < ny - 1 && row >= col)
      { 
        T cosine, sine; const int row2 = row + 1;

        if (l_id == 0)
        {
          T a = M[row * ld_m + col], b = M[row2 * ld_m + col], r, p;

          if (b == 0)
          { cosine = signbit(a) * -2 + 1; sine = 0; r = fabs(a); p = 0; }
          else if (a == 0)
          { cosine = 0; sine = signbit(b) * -2 + 1; r = fabs(b); p = 1; }
          else if (fabs(b) > fabs(a))
          { T t = - a / b; sine = rhypot(1, t); cosine = sine * t; r = - b / sine; p = 2 / cosine; } // rhypot(1, t) = 1 / sqrt(1 + t * t);
          else
          { T t = - b / a; cosine = rhypot(1, t); sine = cosine * t; r = a / cosine; p = sine / 2; }

          M[row * ld_m + col] = r;
          M[row2 * ld_m + col] = p;
        }
        __syncwarp();

        cosine = __shfl_sync(0xffffffff, cosine, 0, warpSize);
        sine = __shfl_sync(0xffffffff, sine, 0, warpSize);

        for (int i = col + l_id + 1; i < nx; i += warpSize)
        {
          const T a = M[row * ld_m + i], b = M[row2 * ld_m + i];
          M[row * ld_m + i] = cosine * a - sine * b;
          M[row2 * ld_m + i] = sine * a + cosine * b;
        }
        __syncwarp();
      }
    }
    __syncthreads();
  }
}

template <class T>
__device__ void blockGivensRecoverQ (T * __restrict__ Q, const T * __restrict__ R, const int nx, const int ny, const int p, const int ld_q, const int ld_r)
{
  const int l_id = lane_rank(), w_id = warp_rank(), tb_w_size = num_warps(), n = nx + ny - 2;

  for (int row = w_id; row < ny; row += tb_w_size) for (int col = l_id; col < p; col += warpSize)
  { Q[row * ld_q + col] = (row == col) ? 1. : 0.; }
  __syncthreads();

  for (int iter = 0; iter < n; iter++)
  {
    for (int col = w_id; col < nx; col += tb_w_size)
    {
      const int row = 2 * col + 1 - nx + iter;
      if (row < ny - 1 && row >= col)
      {
        T cosine, sine; const int row2 = row + 1;

        if (l_id == 0)
        {
          T p = R[row2 * ld_r + col];

          if (p == 0)
          { cosine = 1; sine = 0; }
          else if (p == 1)
          { cosine = 0; sine = 1; }
          else if (fabs(p) > 2)
          { cosine = 2 / p; sine = sqrt(1 - cosine * cosine); }
          else
          { sine = 2 * p; cosine = sqrt(1 - sine * sine); }

        }
        __syncwarp();

        cosine = __shfl_sync(0xffffffff, cosine, 0, warpSize);
        sine = __shfl_sync(0xffffffff, sine, 0, warpSize);

        for (int i = col + l_id; i < p; i += warpSize)
        {
          const T a = Q[row * ld_q + i], b = Q[row2 * ld_q + i];
          Q[row * ld_q + i] = cosine * a + sine * b;
          Q[row2 * ld_q + i] = - sine * a + cosine * b;
        }
        __syncwarp();
      }
    }
    __syncthreads();
  }
}

template <class T, class vecT, int vec_size, int block_dim_m, int block_dim_k>
__device__ void blockLowRankAccum (T * __restrict__ U1, T * __restrict__ VT1, const T * __restrict__ U2, const T * __restrict__ VT2, const int nx, const int ny, 
  const int k1, const int k2, const int ld_u1, const int ld_vt1, const int ld_u2, const int ld_vt2, T * __restrict__ shm)
{
  const int t_id = thread_rank();
  T * U, ** U_ptr = (T **) &shm[0], * V, ** V_ptr = (T **) &shm[1], * Q, ** Q_ptr = (T **) &shm[2];

  if (t_id == 0)
  { * U_ptr = new T[ny * k1]; * V_ptr = new T[nx * k1]; * Q_ptr = new T[ny * k1]; }
  __syncthreads();

  U = *U_ptr; V = *V_ptr; Q = *Q_ptr;
  __syncthreads();

  blockDenseGemm_3x <T, vecT, vec_size, block_dim_m, block_dim_k> (1., 0, U, U1, VT1, dev_rnd_seed, ny, k1, k1, nx, k1, ld_u1, ld_vt1, k1, false, true, false, 0, k1 * k1, shm);
  blockDenseGemm_3x <T, vecT, vec_size, block_dim_m, block_dim_k> (1., 1., U, U2, VT2, dev_rnd_seed, ny, k1, k2, nx, k1, ld_u2, ld_vt2, k1, false, true, false, 0, k2 * k1, shm);

  blockGivensRotation <T> (U, k1, ny, k1);
  blockGivensRecoverQ <T> (Q, U, k1, ny, k1, k1, k1);

  blockDenseGemm_3x <T, vecT, vec_size, block_dim_m, block_dim_k> (1., 0., V, VT1, U1, Q, nx, k1, k1, ny, k1, ld_vt1, ld_u1, k1, false, true, false, 0, k1 * k1, shm);
  blockDenseGemm_3x <T, vecT, vec_size, block_dim_m, block_dim_k> (1., 1., V, VT2, U2, Q, nx, k1, k2, ny, k1, ld_vt2, ld_u2, k1, false, true, false, 0, k2 * k1, shm);

  matrixCopy <T, vecT, vec_size> (V, VT1, k1, nx, k1, ld_vt1, false);
  matrixCopy <T, vecT, vec_size> (Q, U1, k1, ny, k1, ld_u1, false);
  __syncthreads();
  
  if (t_id == 0)
  { delete[] U; delete[] V; delete[] Q; }
  __syncthreads();

}

template <class T>
__device__ int blockReadRank (T * __restrict__ A, const int nx, const int ny, const int ld, const double epi, T * __restrict__ shm, const int shm_size)
{
  const int step = shm_size / nx, total = step * nx;

  for (int i = thread_rank(); i < total; i += block_dim())
  { shm[i] = 0; }
  __syncthreads();

  for (int i = 0; i < ny; i += step)
  {
    for (int j = thread_rank(); j < total; j += block_dim())
    {
      const int row = i + j / nx, col = j - (row - i) * nx;
      const T e = A[row * ld + col];
      shm[j] += e * e;
    }
  }
  __syncthreads();

  int r = 0;
  for (int i = thread_rank(); i < nx; i += block_dim())
  {
    T norm = 0;
    for (int j = 0; j < step; j++)
    { norm += shm[j * nx + i]; }
    r += (int) (norm >= epi);
  }
  __syncthreads();
  
  const int r_ = blockAllReduceSum <int> (r, (int *) shm);
  __syncthreads();

  return r_;
}

template <class T> 
__device__ int blockRandomizedSVD (T * __restrict__ A, T * __restrict__ VT, const int nx, const int ny, const int ld_a, const int ld_v, 
  const int rank, const double epi, const int iter_limit, T * __restrict__ shm, const int shm_size)
{
  /*const int P = rank > nx ? (nx > ny ? ny : nx) : (rank > ny ? ny : rank);

  T * X, ** X_ptr = (T **) &shm[0], *Y, **Y_ptr = (T **) &shm[1], *B, ** B_ptr = (T **) &shm[2];
  if (thread_rank() == 0)
  { 
    X = new T[ny * P]; *X_ptr = X; 
    Y = new T[ny * P]; *Y_ptr = Y; 
    B = new T[P * nx]; *B_ptr = B; 
  }
  __syncthreads();

  X = *X_ptr; Y = *Y_ptr; B = *B_ptr;
  __syncthreads();

  blockDenseGemm_shm (1., 0., X, A, dev_rnd_seed, ny, P, nx, P, ld_a, P, false, false, shm, shm_size);

  matrixCopy_fromRM (X, Y, P, ny, P, P, false);
  blockGivensRotation (X, P, ny, P);
  blockDenseTrsmR_shm (Y, X, P, ny, P, P, P, false, shm, shm_size);
  blockGramSchmidt (Y, P, ny, P, shm);

  blockDenseGemm_shm (1., 0., B, Y, A, P, nx, ny, nx, P, ld_a, true, false, shm, shm_size);*/

  int * iter = (int *) &shm[0], *loop_counter = (int *) &shm[1];
  if (thread_rank() == 0)
  { *iter = 1; *loop_counter = 0; }
  __syncthreads();

  while (*iter && *loop_counter < iter_limit)
  {
    if (thread_rank() == 0)
    { *iter = 0; (*loop_counter)++; }
    __syncthreads();

    bool iter_result = blockSingleSideJacobiSVD(A, VT, nx, ny, ld_a, ld_v, &shm[2], epi);
    if (thread_rank() == 0)
    { *iter = (int) iter_result; }
    __syncthreads();
  }

  /*blockDenseGemm_shm (1., 0., A, Y, B, ny, nx, P, ld_a, P, nx, false, false, shm, shm_size);
  const int r = blockReadRank <T> (B, nx, P, nx, epi, shm, shm_size);
  __syncthreads();

  if (thread_rank() == 0)
  { delete X; delete Y; delete B; }
  __syncthreads();*/

  const int r = blockReadRank <T> (A, nx, ny, nx, epi, shm, shm_size);

  return r;

}


#endif