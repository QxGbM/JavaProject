#ifndef _SVD_CUH
#define _SVD_CUH

#include <pspl.cuh>

void swap_col (double *col1, double *col2, const int ny, const int ld)
{
  for (int i = 0; i < ny; i++)
  {
    const double t = col1[i * ld]; col1[i * ld] = col2[i * ld]; col2[i * ld] = t;
  }
}


int test1 () 
{

  const int nx = 8, ny = 4;
  
  dev_dense <double> *d_VT, *d_A;

  d_A = new dev_dense <double> (nx, ny);
  d_A -> loadTestMatrix();

  d_VT = new dev_dense <double> (nx, nx);
  d_VT -> loadIdentityMatrix();

  double *A = d_A -> getElements();
  double *VT = d_VT -> getElements();

svd:
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
        double t = s_ii; s_ii = s_jj; s_jj = t;
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

  if (iter) goto svd;

  d_A -> print();
  d_VT -> print();

  for (int i = 0; i < nx; i++)
  {
    double s = 0.0;
    for (int j = 0; j < ny; j++)
    { s += A[j * nx + i] * A[j * nx + i]; }

    s = sqrt(s);
    printf("%d: %e\n", i, s);
  }

  dev_dense <double> *c = d_A -> matrixMultiplication(d_VT -> transpose());
  c -> print();

  return 0;
}

#endif