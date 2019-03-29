#ifndef _SVD_CUH
#define _SVD_CUH

#include <pspl.cuh>

#define target 1.e-8

int test1 () 
{

  const int M = 4, N = 4;
  
  dev_dense <double> *d_U, *d_S, *d_VT, *d_A;

  d_U = new dev_dense <double> (M, M);
  d_S = new dev_dense <double> (M, N);

  d_A = new dev_dense <double> (M, N);
  d_A -> loadTestMatrix();

  d_VT = new dev_dense <double> (N, N);
  d_VT -> loadIdentityMatrix();

  double *A = d_A->getElements();
  double *S = d_S->getElements();
  double *U = d_U->getElements();
  double *VT = d_VT->getElements();

  double convergence;
  do
  {
    convergence = 0.0;	
    for(int i = 1; i < M; i++)
    {
      for(int j = 0; j < i; j++)
      {
        // Refer to: https://blog.csdn.net/k531623594/article/details/50628163

        double s_ii = 0.0;
        double s_jj = 0.0;
        double s_ij = 0.0;

        for(int k = 0; k < N ; k++)
        {
          s_ii += A[i * M + k] * A[k * M + i]; // s_ii = A[i]_T (horizontal) * A[i] (orthogonal)
          s_jj += A[j * M + k] * A[k * M + j]; // s_jj = A[j]_T (horizontal) * A[j] (orthogonal)
          s_ij += A[i * M + k] * A[k * M + j]; // s_ij = A[i]_T (horizontal) * A[j] (orthogonal)
        }

        double t2 = (s_jj - s_ii) / 2 * s_ij; // torque^2
        double t = 1.0 / (abs(t2) + sqrt(1.0 + t2 * t2)); // tangent
        double c = 1.0 / (sqrt(1.0 + (t * t))); // cosine
        double s = c * t; // sine

        if (t2 < 0) { s = -s; }

        for(int k = 0; k < N; k++) //Apply rotations on U and V
        {
          double ai_T = A[i * M + k], aj_T = A[j * M + k]; // A[i]_T (horizontal), A[j]_T (horizontal)
          A[i * M + k] = c * ai_T - s * aj_T;
          A[j * M + k] = s * ai_T + c * aj_T;

          double vi_T = VT[i * M + k], vj_T = VT[j * M + k]; // V[i]_T (horizontal), V[j]_T (horizontal)
          VT[i * M + k] = c * vi_T - s * vj_T;
          VT[j * M + k] = s * vi_T + c * vj_T;
        }

        double new_convergence = abs(s_ij) / sqrt(s_ii * s_jj);

        convergence = (convergence > new_convergence) ? convergence : new_convergence;

      }
    }
  } while(convergence > target);


  for(int i = 0; i < M; i++)
  {
    double t = 0;
    for(int j = 0; j < N; j++)
    {
      t += A[i * M + j] * A[i * M + j];
    }
    t = sqrt(t);

    S[i * M + i] = t;

    for(int j = 0; j < N; j++)
    {
      U[j * M + i] = A[i * M + j] / t;
    }
  }

  d_U -> print();
  d_S -> print();
  d_VT -> print();

  dev_dense <double> *c = d_U -> matrixMultiplication(d_S) -> matrixMultiplication(d_VT);
  c -> print();

  return 0;
}

#endif