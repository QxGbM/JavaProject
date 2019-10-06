
#pragma once
#ifndef _DEV_LOW_RANK_FUNCS_CUH
#define _DEV_LOW_RANK_FUNCS_CUH

#include <pspl.cuh>


DEVICE int warpAllReduceSum (int value)
{
  for (int mask = warpSize / 2; mask > 0; mask /= 2)
  { value += __shfl_xor_sync(0xffffffff, value, mask, warpSize); }
  return value;
}

DEVICE int blockAllReduceSum (int value, int * shm)
{
  value = warpAllReduceSum (value);

  if (lane_rank() == 0)
  { shm[warp_rank()] = value; }
  __syncthreads();

  if (block_dim() > warpSize && warp_rank() == 0)
  {
    value = 0;
    for (int i = lane_rank(); i < num_warps(); i += warpSize)
    { value += shm[i]; }

    value = warpAllReduceSum (value);
    if (lane_rank() == 0)
    { shm[0] = value; }
  }
  __syncthreads();

  return shm[0];
}


DEVICE bool blockSingleSideJacobiSVD (real_t * __restrict__ UxS, real_t * __restrict__ VT, const int nx, const int ny, const int ld_UxS, const int ld_VT, real_t * __restrict__ shm, const real_t epi)
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
        const real_t e = UxS[row_UxS + col];
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
          const real_t e1 = swap ? UxS[row_UxS + col2] : UxS[row_UxS + col], e2 = swap ? UxS[row_UxS + col] : UxS[row_UxS + col2];
          UxS[row_UxS + col] = cosine * e1 - sine * e2;
          UxS[row_UxS + col2] = sine * e1 + cosine * e2;
        }

        for (int row = lane_rank(); row < nx; row += warpSize)
        {
          const int row_Vreal_t = row * ld_VT;
          const real_t e3 = swap ? VT[row_Vreal_t + col2] : VT[row_Vreal_t + col], e4 = swap ? VT[row_Vreal_t + col] : VT[row_Vreal_t + col2];
          VT[row_Vreal_t + col] = cosine * e3 - sine * e4;
          VT[row_Vreal_t + col2] = sine * e3 + cosine * e4;
        }
        __syncwarp();

      }
    }
    __syncthreads();
  }

  return iter;
}


DEVICE void blockGivensRotation (real_t * __restrict__ M, const int nx, const int ny, const int ld_m)
{
  const int l_id = lane_rank(), w_id = warp_rank(), n_wp = num_warps(), n = nx + ny - 2;

  for (int iter = 0; iter < n; iter++)
  {
    for (int col = w_id; col < nx; col += n_wp)
    {
      const int row = ny - 2 + 2 * col - iter;
      if (row < ny - 1 && row >= col)
      {
        const int row2 = row + 1;
        real_t cosine, sine, * M_row = &M[row * ld_m], * M_row2 = &M[row2 * ld_m]; 

        if (l_id == 0)
        {
          real_t a = M_row[col], b = M_row2[col], r, p;

          if (b == 0)
          { cosine = copysign(1., a); sine = 0.; r = fabs(a); p = 0.; }
          else if (a == 0)
          { cosine = 0.; sine = copysign(1., b); r = fabs(b); p = 1.; }
          else if (fabs(b) > fabs(a))
          { real_t t = - a / b; sine = rhypot(1., t); cosine = sine * t; r = - b / sine; p = 2. / cosine; } // rhypot(1, t) = 1 / sqrt(1 + t * t);
          else
          { real_t t = - b / a; cosine = rhypot(1., t); sine = cosine * t; r = a / cosine; p = sine / 2.; }

          M_row[col] = r;
          M_row2[col] = p;
        }

        cosine = __shfl_sync(0xffffffff, cosine, 0, warpSize);
        sine = __shfl_sync(0xffffffff, sine, 0, warpSize);

        for (int i = col + l_id + 1; i < nx; i += warpSize)
        {
          const real_t a = M_row[i], b = M_row2[i];
          M_row[i] = fma (cosine, a, fma (- sine, b, 0.)); // cosine * a - sine * b;
          M_row2[i] = fma (sine, a, fma (cosine, b, 0.)); // sine * a + cosine * b;
        }
      }
    }
    __syncthreads();
  }
}

DEVICE void blockGivensRecoverQ (real_t * __restrict__ Q, const real_t * __restrict__ R, const int nx, const int ny, const int p, const int ld_q, const int ld_r)
{
  const int l_id = lane_rank(), w_id = warp_rank(), n_wp = num_warps(), n = nx + ny - 2;

  for (int row = w_id; row < ny; row += n_wp) for (int col = l_id; col < p; col += warpSize)
  { Q[row * ld_q + col] = (real_t) (row == col); }
  __syncthreads();

  for (int iter = 0; iter < n; iter++)
  {
    for (int col = w_id; col < nx; col += n_wp)
    {
      const int row = 2 * col + 1 - nx + iter;
      if (row < ny - 1 && row >= col)
      {
        const int row2 = row + 1;
        real_t cosine, sine, * Q_row = &Q[row * ld_q], * Q_row2 = &Q[row2 * ld_q];

        if (l_id == 0)
        {
          real_t p = R[row2 * ld_r + col];

          if (p == 0)
          { cosine = 1.; sine = 0.; }
          else if (p == 1)
          { cosine = 0.; sine = 1.; }
          else if (fabs(p) > 2)
          { cosine = 2. / p; sine = sqrt (fma (cosine, - cosine, 1.)); }
          else
          { sine = 2. * p; cosine = sqrt (fma (sine, - sine, 1.)); }

        }

        cosine = __shfl_sync(0xffffffff, cosine, 0, warpSize);
        sine = __shfl_sync(0xffffffff, sine, 0, warpSize);

        for (int i = col + l_id; i < p; i += warpSize)
        {
          const real_t a = Q_row[i], b = Q_row2[i];
          Q_row[i] = fma (cosine, a, fma (sine, b, 0.)); // cosine * a + sine * b;
          Q_row2[i] = fma (- sine, a, fma (cosine, b, 0.)); // - sine * a + cosine * b;
        }

      }
    }
    __syncthreads();
  }
}

DEVICE void blockLowRankAccum (real_t * __restrict__ U1, real_t * __restrict__ VT1, const real_t * __restrict__ U2, const real_t * __restrict__ VT2, const int nx, const int ny, const int k1, const int k2, 
  const int ld_u1, const int ld_vt1, const int ld_u2, const int ld_vt2, const int offset1, const int offset2, real_t * __restrict__ shm, real_t * __restrict__ my_tmp, const real_t *__restrict__ rnd_seed)
{
  real_t * U = my_tmp, * V = &my_tmp[offset1], * Q = &my_tmp[offset2];

  blockDenseGemm (1., 0., Q, rnd_seed, VT1, k1, k1, nx, k1, k1, ld_vt1, true, false, shm);
  blockDenseGemm (1., 0., U, U1, Q, ny, k1, k1, k1, ld_u1, k1, false, true, shm);

  blockDenseGemm (1., 0., Q, rnd_seed, VT2, k1, k2, nx, k2, k1, ld_vt2, true, false, shm);
  blockDenseGemm (1., 1., U, U2, Q, ny, k1, k2, k1, ld_u2, k2, false, true, shm);

  blockGivensRotation (U, k1, ny, k1);
  blockGivensRecoverQ (Q, U, k1, ny, k1, k1, k1);

  blockDenseGemm (1., 0., U, Q, U1, k1, k1, ny, k1, k1, ld_u1, true, false, shm);
  blockDenseGemm (1., 0., V, VT1, U, nx, k1, k1, k1, ld_vt1, k1, false, true, shm);

  matrixCopy (Q, U1, k1, ny, k1, ld_u1);

  blockDenseGemm (1., 0., U, Q, U2, k1, k2, ny, k2, k1, ld_u2, true, false, shm);
  blockDenseGemm (1., 1., V, VT2, U, nx, k1, k2, k1, ld_vt2, k2, false, true, shm);

  matrixCopy (V, VT1, k1, nx, k1, ld_vt1);
  __syncthreads();

}

DEVICE int blockReadRank (real_t * __restrict__ A, const int nx, const int ny, const int ld, const double epi, real_t * __restrict__ shm, const int shm_size)
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
      const real_t e = A[row * ld + col];
      shm[j] += e * e;
    }
  }
  __syncthreads();

  int r = 0;
  for (int i = thread_rank(); i < nx; i += block_dim())
  {
    real_t norm = 0;
    for (int j = 0; j < step; j++)
    { norm += shm[j * nx + i]; }
    r += (int) (norm >= epi);
  }
  __syncthreads();
  
  const int r_ = blockAllReduceSum (r, (int *) shm);
  __syncthreads();

  return r_;
}

DEVICE int blockRandomizedSVD (real_t * __restrict__ A, real_t * __restrict__ VT, const int nx, const int ny, const int ld_a, const int ld_v, 
  const int rank, const double epi, const int iter_limit, real_t * __restrict__ shm, const int shm_size)
{
  /*const int P = rank > nx ? (nx > ny ? ny : nx) : (rank > ny ? ny : rank);

  real_t * X, ** X_ptr = (real_t **) &shm[0], *Y, **Y_ptr = (real_t **) &shm[1], *B, ** B_ptr = (real_t **) &shm[2];
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

  const int r = blockReadRank (A, nx, ny, nx, epi, shm, shm_size);

  return r;

}


#endif