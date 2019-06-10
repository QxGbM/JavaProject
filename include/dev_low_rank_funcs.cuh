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

template <class T> __device__ double sumSquaredDifference (const T * A, const T * B, const int nx, const int ny, const int ld_a, const int ld_b, T * shm)
{
  double thread_sum = 0.;
  for (int i = thread_rank(); i < nx * ny; i += block_dim())
  { 
    const int row = i / nx, col = i - row * nx;
    const double diff = A[row * ld_a + col] * B[row * ld_b + col];
    thread_sum += diff * diff; 
  }

  return blockAllReduceSum <double> (thread_sum, (double *) shm);
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


template <class T> __device__ void loadIdentity (T * M, const int nx, const int ny, const int ld)
{
  for (int i = thread_rank(); i < nx * ny; i++)
  {
    const int row = i / nx, col = i - row * nx;
    M[row * ld + col] = (T) (row == col);
  }
  __syncthreads();
}

template <class T> __device__ void blockGramSchmidt (T * __restrict__ M, const int nx, const int ny, const int ld, T * __restrict__ shm)
{
  for (int i = 0; i < nx; i++)
  {
    for (int col = i + thread_rank(); col < nx; col += block_dim())
    { shm[col] = 0; }
    __syncthreads();

    for (int row = 0; row < ny; row++) for (int col = i + thread_rank(); col < nx; col += block_dim())
    { shm[col] += M[row * ld + i] * M[row * ld + col]; }
    __syncthreads();

    if (thread_rank() == 0)
    { shm[i] = rsqrt(shm[i]); }
    __syncthreads();

    for (int col = i + thread_rank() + 1; col < nx; col += block_dim())
    { shm[col] *= shm[i]; }
    __syncthreads();

    for (int row = 0; row < ny; row++) 
    {
      if (thread_rank() == 0)
      { shm[nx] = M[row * ld + i] * shm[i]; }
      __syncthreads();

      if (thread_rank() == 0)
      { M[row * ld + i] = shm[nx]; }
      for (int col = i + thread_rank() + 1; col < nx; col += block_dim())
      { M[row * ld + col] = M[row * ld + col] - shm[col] * shm[nx]; }
      __syncthreads();
    }
  }

}

template <class T> 
__device__ void blockGivensRotation (T * __restrict__ M, const int nx, const int ny, const int ld)
{
  const int n = nx >= ny ? ny - 1 : nx;

  for (int i = 0; i < n; i++)
  {
    const int cols = ny - i;
    int last_step = cols;

    for (int step = cols / 2; step > 0; step /= 2)
    {
      const int rows_step = i + step, rows_total = i + last_step;

      for (int row = i + warp_rank(); row < rows_step; row += num_warps())
      {
        const int row2 = row + step, row3 = row + step + step;

        if (row3 < rows_total)
        {
          T cosine1, sine1, cosine2, sine2;
          
          if (lane_rank() == 0)
          {
            T a = M[row * ld + i], b = M[row2 * ld + i], c = M[row3 * ld + i], r;

            if (b == 0)
            { cosine1 = signbit(a) * -2 + 1; sine1 = 0; r = fabs(a); }
            else if (a == 0)
            { cosine1 = 0; sine1 = signbit(b) * -2 + 1; r = fabs(b); }
            else if (fabs(b) > fabs(a))
            { T t = a / b, st = sqrt(1 + t * t); sine1 = (signbit(b) * 2 - 1) / st; cosine1 = (-sine1) * t; r = fabs(b) * st; }
            else
            { T t = b / a, st = sqrt(1 + t * t); cosine1 = (signbit(a) * -2 + 1) / st; sine1 = (-cosine1) * t; r = fabs(a) * st; }

            if (c == 0)
            { cosine2 = 1; sine2 = 0; }
            else if (r == 0)
            { cosine2 = 0; sine2 = signbit(c) * -2 + 1; r = fabs(c); }
            else if (fabs(c) > r)
            { T t = r / c; sine2 = (signbit(c) * 2 - 1) * rsqrt(1. + t * t); cosine2 = (-sine2) * t; r = fabs(c) * sqrt(1. + t * t); }
            else
            { T t = c / r; cosine2 = rsqrt(1. + t * t); sine2 = (-cosine2) * t; r = r * sqrt(1. + t * t); }

            M[row * ld + i] = r;
            M[row2 * ld + i] = 0;
            M[row3 * ld + i] = 0;
          }
          __syncwarp();

          cosine1 = __shfl_sync(0xffffffff, cosine1, 0, warpSize);
          sine1 = __shfl_sync(0xffffffff, sine1, 0, warpSize);
          cosine2 = __shfl_sync(0xffffffff, cosine2, 0, warpSize);
          sine2 = __shfl_sync(0xffffffff, sine2, 0, warpSize);

          for (int col = i + lane_rank() + 1; col < nx; col += warpSize)
          {
            const T a = M[row * ld + col], b = M[row2 * ld + col], c = M[row3 * ld + col];
            const T r = cosine1 * a - sine1 * b;
            M[row * ld + col] = cosine2 * r - sine2 * c;
            M[row2 * ld + col] = sine1 * a + cosine1 * b;
            M[row3 * ld + col] = sine2 * r + cosine2 * c;
          }
          __syncwarp();
        }
        else
        {
          T cosine, sine;

          if (lane_rank() == 0)
          {
            T a = M[row * ld + i], b = M[row2 * ld + i], r;

            if (b == 0)
            { cosine = signbit(a) * -2 + 1; sine = 0; r = fabs(a); }
            else if (a == 0)
            { cosine = 0; sine = signbit(b) * -2 + 1; r = fabs(b); }
            else if (fabs(b) > fabs(a))
            { T t = a / b, st = sqrt(1 + t * t); sine = (signbit(b) * 2 - 1) / st; cosine = (-sine) * t; r = fabs(b) * st; }
            else
            { T t = b / a, st = sqrt(1 + t * t); cosine = (signbit(a) * -2 + 1) / st; sine = (-cosine) * t; r = fabs(a) * st; }

            M[row * ld + i] = r;
            M[row2 * ld + i] = 0;
          }
          __syncwarp();

          cosine = __shfl_sync(0xffffffff, cosine, 0, warpSize);
          sine = __shfl_sync(0xffffffff, sine, 0, warpSize);

          for (int col = i + lane_rank() + 1; col < nx; col += warpSize)
          {
            const T a = M[row * ld + col], b = M[row2 * ld + col];
            M[row * ld + col] = cosine * a - sine * b;
            M[row2 * ld + col] = sine * a + cosine * b;
          }
          __syncwarp();
        }
      }
      last_step = step;
      __syncthreads();

    }
  }
}

template <class T> 
__device__ void blockLowRankAccum (T * __restrict__ U1, T * __restrict__ VT1, const T * __restrict__ U2, const T * __restrict__ VT2, const int nx, const int ny, 
  const int k1, const int k2, const int ld_u1, const int ld_vt1, const int ld_u2, const int ld_vt2, T * __restrict__ shm, const int shm_size)
{
  T * U, ** U_ptr = (T **) &shm[0], * V, ** V_ptr = (T **) &shm[1], * Q, ** Q_ptr = (T **) &shm[2];
  if (thread_rank() == 0)
  {
    U = new T[ny * k1]; *U_ptr = U;
    V = new T[nx * k1]; *V_ptr = V;
    Q = new T[ny * k1]; *Q_ptr = Q;
  }
  __syncthreads();

  U = *U_ptr; V = *V_ptr; Q = *Q_ptr;
  __syncthreads();

  blockDenseGemm_3x_shm <T> (1., 0, U, U1, VT1, dev_rnd_seed, ny, k1, k1, nx, k1, ld_u1, ld_vt1, k1, false, true, false, shm, shm_size);
  __syncthreads();

  blockDenseGemm_3x_shm <T> (1., 1., U, U2, VT2, dev_rnd_seed, ny, k1, k2, nx, k1, ld_u2, ld_vt2, k1, false, true, false, shm, shm_size);
  __syncthreads();

  matrixCopy_fromRM <T> (U, Q, k1, ny, k1, k1, false);
  __syncthreads();

  blockGivensRotation <T> (U, k1, ny, k1);
  __syncthreads();

  blockDenseTrsmR_shm <T> (Q, U, k1, ny, k1, k1, k1, false, shm, shm_size);
  __syncthreads();

  blockGramSchmidt <T> (Q, k1, ny, k1, shm);
  __syncthreads();

  blockDenseGemm_3x_shm <T> (1., 0., V, VT1, U1, Q, nx, k1, k1, ny, k1, ld_vt1, ld_u1, k1, false, true, false, shm, shm_size);
  __syncthreads();

  blockDenseGemm_3x_shm <T> (1., 1., V, VT2, U2, Q, nx, k1, k2, ny, k1, ld_vt2, ld_u2, k1, false, true, false, shm, shm_size);
  __syncthreads();

  matrixCopy_fromRM <T> (V, VT1, k1, nx, k1, ld_vt1, false);
  matrixCopy_fromRM <T> (Q, U1, k1, ny, k1, ld_u1, false);
  __syncthreads();
  
  if (thread_rank() == 0)
  { delete U; delete V; delete Q; }
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


template <class T>
/* Convert LR to dense by multiplying U with VT. VT is reset to Identity matrix and rank is set to -1. */
__device__ void blockLrToDense (T * __restrict__ U, T * __restrict__ VT, int * __restrict__ rank, const int nx, const int ny,
  const int ld_u, const int ld_vt, T * __restrict__ shm, const int shm_size)
{
  const int step_size = shm_size / nx, rank_old = *rank;

  for (int row = 0; row < ny; row += step_size)
  {
    const int rows = (ny - row > step_size) ? step_size : ny - row;
    blockDenseGemm_shm <T> (1., 0., shm, &U[row * ld_u], VT, rows, nx, rank_old, nx, ld_u, ld_vt, false, true, nullptr, 0);
    matrixCopy_toRM <T> (shm, &U[row * ld_u], nx, rows, nx, ld_u, false);
    __syncthreads();
  }

  if (thread_rank() == 0)
  { *rank = -1; }
  loadIdentity (VT, nx, nx, ld_vt);
  __syncthreads();
}


template <class T>
/* LR = alpha * D * D + beta * old. LR is converted to Dense. */
__device__ void blockGemm_lr_d_d_shm (const T alpha, const T beta, T * __restrict__ U, T * __restrict__ VT, int * __restrict__ rank, 
  const int nx, const int ny, const int ld_u, const int ld_vt, const int k, const T * __restrict__ A, const int ld_a, const bool a_T, 
  const T * __restrict__ B, const int ld_b, const bool b_T, T * __restrict__ shm, const int shm_size)
{
  if (*rank >= 0) 
  { blockLrToDense <T> (U, VT, rank, nx, ny, ld_u, ld_vt, shm, shm_size); }
  blockDenseGemm_shm <T> (alpha, beta, U, A, B, ny, nx, k, ld_u, ld_a, ld_b, a_T, b_T, shm, shm_size);
}

template <class T>
/* LR = alpha * D * LR + beta * old. Assumes alpha is not 0. */
__device__ void blockGemm_lr_d_lr_shm (const T alpha, const T beta, T * __restrict__ U, T * VT, int * rank, const int nx, const int ny, 
  const int ld_u, const int ld_vt, const int k, const T * __restrict__ A, const int ld_a, const bool a_T, 
  const T * __restrict__ U_B, const T * VT_B, const int * rank_b, const int ld_ub, const int ld_vtb, T * __restrict__ shm, const int shm_size)
{
  const int rank_b_ = *rank_b, rank_old = *rank, rank_ = rank_old + rank_b_;
  __syncthreads();

  if (rank_ == -2) /* Both LR are Denses. -> Normal Dense GEMM on Us. */
  { 
    blockDenseGemm_shm <T> (alpha, beta, U, A, U_B, ny, nx, k, ld_u, ld_a, ld_ub, a_T, false, shm, shm_size); 
  }
  else if (rank_b_ == -1) /* B is dense & M is LR. -> Convert M to dense and do normal GEMM. */
  { 
    blockLrToDense <T> (U, VT, rank, nx, ny, ld_u, ld_vt, shm, shm_size);
    blockDenseGemm_shm <T> (alpha, beta, U, A, U_B, ny, nx, k, ld_u, ld_a, ld_ub, a_T, false, shm, shm_size);
  }
  else if (beta == 0.) /* M is reset & B is LR. -> M has same rank as B. */
  {
    if (thread_rank() == 0) 
    { *rank = rank_b_; }
    matrixCopy_fromRM <T> (VT_B, VT, rank_b_, nx, ld_vtb, ld_vt, false);
    blockDenseGemm_shm <T> (alpha, 0., U, A, U_B, ny, rank_b_, k, ld_u, ld_a, ld_ub, a_T, false, shm, shm_size);
  }
  else if (rank_old == -1) /* M is dense, not being reset & B is LR. -> update M using A x U_B x V_B. */
  { 
    blockDenseGemm_3x_shm <T> (alpha, beta, U, A, U_B, VT_B, ny, nx, k, rank_b_, ld_u, ld_a, ld_ub, ld_vtb, a_T, false, true, shm, shm_size); 
  }
  else if (VT == VT_B && rank == rank_b) /* M and B are LR, shared vertical basis and rank. -> Update U using A * U_B. */
  { 
    blockDenseGemm_shm <T> (alpha, beta, U, A, U_B, ny, rank_old, k, ld_u, ld_a, ld_ub, a_T, false, shm, shm_size); 
  }
  else if (rank_ >= nx || rank_ >= ny) /* M + B exceeds rank. -> Convert M to dense and do normal GEMM. */
  {
    blockLrToDense <T> (U, VT, rank, nx, ny, ld_u, ld_vt, shm, shm_size);
    blockDenseGemm_3x_shm <T> (alpha, beta, U, A, U_B, VT_B, ny, nx, k, rank_b_, ld_u, ld_a, ld_ub, ld_vtb, a_T, false, true, shm, shm_size);
  }
  else /* M and B LR, M not being reset, and M maintains a LR. -> Concatenate U and V. */
  {
    if (thread_rank() == 0) 
    { *rank = rank_; }
    blockDenseScalar <T> (beta, U, ny, rank_old, ld_u);
    matrixCopy_fromRM <T> (VT_B, &VT[rank_old], rank_b_, nx, ld_vtb, ld_vt, false);
    blockDenseGemm_shm <T> (alpha, 0., &U[rank_old], A, U_B, ny, rank_b_, k, ld_u, ld_a, ld_ub, a_T, false, shm, shm_size);
  }
}

template <class T>
/* LR = alpha * LR * D + beta * old. Assumes alpha is not 0. */
__device__ void blockGemm_lr_lr_d_shm(const T alpha, const T beta, T * U, T * __restrict__ VT, int * rank, const int nx, const int ny, 
  const int ld_u, const int ld_vt, const int k, const T * U_A, const T * __restrict__ VT_A, const int * rank_a, const int ld_ua, const int ld_vta, 
  const T * __restrict__ B, const int ld_b, const bool b_T, T * __restrict__ shm, const int shm_size)
{
  const int rank_a_ = *rank_a, rank_old = *rank, rank_ = rank_old + rank_a_;
  __syncthreads();

  if (rank_ == -2) /* Both LR are Denses. -> Normal Dense GEMM on Us. */
  { 
    blockDenseGemm_shm <T> (alpha, beta, U, U_A, B, ny, nx, k, ld_u, ld_ua, ld_b, false, b_T, shm, shm_size); 
  }
  else if (rank_a_ == -1) /* A is dense & M is LR. -> Convert M to dense and do normal GEMM. */
  { 
    blockLrToDense <T> (U, VT, rank, nx, ny, ld_u, ld_vt, shm, shm_size);
    blockDenseGemm_shm <T> (alpha, beta, U, U_A, B, ny, nx, k, ld_u, ld_ua, ld_b, false, b_T, shm, shm_size);
  }
  else if (beta == 0.) /* M is reset & A is LR. -> M has same rank as A. */
  {
    if (thread_rank() == 0) 
    { *rank = rank_a_; }
    matrixCopy_fromRM <T> (U_A, U, rank_a_, ny, ld_ua, ld_u, false);
    blockDenseGemm_shm <T> (alpha, 0., VT, B, VT_A, nx, rank_a_, k, ld_vt, ld_b, ld_vta, !b_T, false, shm, shm_size);
  }
  else if (rank_old == -1) /* M is dense, not being reset & A is LR. -> update M using U_A x V_A x B. */
  { 
    blockDenseGemm_3x_shm <T> (alpha, beta, U, U_A, VT_A, B, ny, nx, rank_a_, k, ld_u, ld_ua, ld_vta, ld_b, false, true, b_T, shm, shm_size); 
  }
  else if (U == U_A && rank == rank_a) /* M and A are LR, shared horizontal basis and rank. -> Update VT using BT * VT_A. */
  { 
    blockDenseGemm_shm <T> (alpha, beta, VT, B, VT_A, nx, rank_old, k, ld_vt, ld_b, ld_vta, !b_T, false, shm, shm_size); 
  }
  else if (rank_ >= nx || rank_ >= ny) /* M + A exceeds rank. -> Convert M to dense and do normal GEMM. */
  {
    blockLrToDense <T> (U, VT, rank, nx, ny, ld_u, ld_vt, shm, shm_size);
    blockDenseGemm_3x_shm <T> (alpha, beta, U, U_A, VT_A, B, ny, nx, rank_a_, k, ld_u, ld_ua, ld_vta, ld_b, false, true, b_T, shm, shm_size); 
  }
  else /* M and A LR, M not being reset, and M maintains a LR. -> Concatenate U and V. */
  {
    if (thread_rank() == 0) 
    { *rank = rank_; }
    blockDenseScalar <T> (beta, VT, nx, rank_old, ld_vt);
    matrixCopy_fromRM <T> (U_A, &U[rank_old], rank_a_, ny, ld_ua, ld_u, false);
    blockDenseGemm_shm <T> (alpha, 0., &VT[rank_old], B, VT_A, nx, rank_a_, k, ld_vt, ld_b, ld_vta, !b_T, false, shm, shm_size);
  }
}

template <class T>
/* LR = alpha * LR * LR + beta * old. Assumes alpha is not 0. */
__device__ void blockGemm_lr_lr_lr_shm (const T alpha, const T beta, T * U, T * VT, int * rank, const int nx, const int ny, const int ld_u, const int ld_vt, 
  const int k, const T * U_A, const T * __restrict__ VT_A, const int * rank_a, const int ld_ua, const int ld_vta, const T * __restrict__ U_B, const T * VT_B, 
  const int * rank_b, const int ld_ub, const int ld_vtb, T * __restrict__ shm, const int shm_size)
{
  const int rank_a_ = *rank_a, rank_b_ = *rank_b, rank_old = *rank;
  __syncthreads();

  if (rank_a_ == -1) /* A is dense. -> do GEMM LR-D-LR. */
  { 
    blockGemm_lr_d_lr_shm <T> (alpha, beta, U, VT, rank, nx, ny, ld_u, ld_vt, k, U_A, ld_ua, false, U_B, VT_B, rank_b, ld_ub, ld_vtb, shm, shm_size); 
  }
  else if (rank_b_ == -1) /* B is dense. -> do GEMM LR-LR-D. */
  { 
    blockGemm_lr_lr_d_shm <T> (alpha, beta, U, VT, rank, nx, ny, ld_u, ld_vt, k, U_A, VT_A, rank_a, ld_ua, ld_vta, U_B, ld_ub, false, shm, shm_size); 
  }
  else if (beta == 0.) /* M is reset & A, B are LR. -> M has same rank as smaller. */
  {
    if (rank_a_ < rank_b_)
    {
      if (thread_rank() == 0) 
      { *rank = rank_a_; }
      matrixCopy_fromRM <T> (U_A, U, rank_a_, ny, ld_ua, ld_u, false);
      blockDenseGemm_3x_shm <T> (alpha, 0., VT, VT_B, U_B, VT_A, nx, rank_a_, rank_b_, k, ld_vt, ld_vtb, ld_ub, ld_vta, false, true, false, shm, shm_size);
    }
    else
    {
      if (thread_rank() == 0) 
      { *rank = rank_b_; }
      matrixCopy_fromRM <T> (VT_B, VT, rank_b_, nx, ld_vtb, ld_vt, false);
      blockDenseGemm_3x_shm <T> (alpha, 0., U, U_A, VT_A, U_B, ny, rank_b_, rank_a_, k, ld_u, ld_ua, ld_vta, ld_ub, false, true, false, shm, shm_size);
    }
  }
  else if (rank_old == -1) /* M is dense, not being reset & A, B are LR. -> update M using U_A x V_A x U_B x V_B. */
  { 
    blockDenseGemm_4x_shm <T> (alpha, beta, U, U_A, VT_A, U_B, VT_B, ny, nx, rank_a_, k, rank_b_, ld_u, ld_ua, ld_vta, ld_ub, ld_vtb, false, true, false, true, shm, shm_size); 
  }
  else if (VT == VT_B && rank == rank_b) /* M and B are LR, shared vertical basis and rank. -> Update U using U_A x V_A * U_B. */
  { 
    blockDenseGemm_3x_shm <T> (alpha, beta, U, U_A, VT_A, U_B, ny, rank_old, rank_a_, k, ld_u, ld_ua, ld_vta, ld_ub, false, true, false, shm, shm_size); 
  }
  else if (U == U_A && rank == rank_a) /* M and B are LR, shared horizontal basis and rank. -> Update VT using VT_B x UT_B * VT_A. */
  {
    blockDenseGemm_3x_shm <T> (alpha, beta, VT, VT_B, U_B, VT_A, nx, rank_old, rank_b_, k, ld_vt, ld_vtb, ld_ub, ld_vta, false, true, false, shm, shm_size); 
  }
  else /* M, A, B all LR, M not being reset. -> Concatenate U and V or convert M to dense */
  {
    if (rank_a_ < rank_b_) /* A has smaller rank. */
    {
      const int rank_ = rank_a_ + rank_old;
      if (rank_ >= nx || rank_ >= ny)
      {
        blockLrToDense <T> (U, VT, rank, nx, ny, ld_u, ld_vt, shm, shm_size);
        blockDenseGemm_4x_shm <T> (alpha, beta, U, U_A, VT_A, U_B, VT_B, ny, nx, rank_a_, k, rank_b_, ld_u, ld_ua, ld_vta, ld_ub, ld_vtb, false, true, false, true, shm, shm_size);
      }
      else
      {
        if (thread_rank() == 0) 
        { *rank = rank_; }
        blockDenseScalar <T> (beta, VT, nx, rank_old, ld_vt);
        matrixCopy_fromRM <T> (U_A, &U[rank_old], rank_a_, ny, ld_ua, ld_u, false);
        blockDenseGemm_3x_shm <T> (alpha, 0., &VT[rank_old], VT_B, U_B, VT_A, nx, rank_a_, rank_b_, k, ld_vt, ld_vtb, ld_ub, ld_vta, false, true, false, shm, shm_size);      
      }
    }
    else /* B has smaller rank. */
    {
      const int rank_ = rank_b_ + rank_old;
      if (rank_ >= nx || rank_ >= ny)
      {
        blockLrToDense <T> (U, VT, rank, nx, ny, ld_u, ld_vt, shm, shm_size);
        blockDenseGemm_4x_shm <T> (alpha, beta, U, U_A, VT_A, U_B, VT_B, ny, nx, rank_a_, k, rank_b_, ld_u, ld_ua, ld_vta, ld_ub, ld_vtb, false, true, false, true, shm, shm_size); 
      }
      else
      {
        if (thread_rank() == 0) 
        { *rank = rank_; }
        blockDenseScalar <T> (beta, U, ny, rank_old, ld_u);
        matrixCopy_fromRM <T> (VT_B, &VT[rank_old], rank_b_, nx, ld_vtb, ld_vt, false);
        blockDenseGemm_3x_shm <T> (alpha, 0., &U[rank_old], U_A, VT_A, U_B, ny, rank_b_, rank_a_, k, ld_u, ld_ua, ld_vta, ld_ub, false, true, false, shm, shm_size);
      }
    }



  }
}

#endif