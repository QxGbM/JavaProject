
#include <sys/time.h>

#include <helper_functions.h>

#define epsilon 1.e-8

int main (int argc, char* argv[]) {

  const int M = 4, N = 4;
  
  double *U, *S, *VT, *A;

  U = zeroMatrix(M, M);
  S = zeroMatrix(M, N);

  A = testMatrix(M, N);
  VT = identityMatrix(N, N);

  timeval start, end;
  gettimeofday(&start, NULL);

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

        double t2 = (s_jj - s_ii) / 2 * s_ij; // torque
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

        convergence = max_(convergence, abs(s_ij) / sqrt(s_ii * s_jj));

      }
    }
  } while(convergence > epsilon);


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

  gettimeofday(&end, NULL);

  double elapsedTime;

  elapsedTime = (end.tv_sec - start.tv_sec) * 1000.0;
  elapsedTime += (end.tv_usec - start.tv_usec) / 1000.0;
  printf("Time: %f ms.\n", elapsedTime);

  printMatrix(U, M, M, N);
  printMatrix(S, M, M, N);
  printMatrix(VT, M, M, N);

  double *b = matrixMultiplication(U, S, M, M, M, M, N);
  double *c = matrixMultiplication(b, VT, M, M, M, M, N);

  printMatrix(c, M, M, N);

  return 0;
}
