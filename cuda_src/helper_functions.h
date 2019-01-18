#ifndef _HELPER_FUNCS
#define _HELPER_FUNCS

#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <memory.h>

#define max(a, b) (a > b) ? a : b
#define min(a, b) (a < b) ? a : b

typedef struct {
  double *head;
  const unsigned nx;
  const unsigned ld;
  const unsigned ny;
} Matrix;

double* testMatrix(const unsigned nx, const unsigned ny);

double* identityMatrix(const unsigned nx, const unsigned ny);

double* randomMatrix(const unsigned nx, const unsigned ny, const double min, const double max);

double* zeroMatrix(const unsigned nx, const unsigned ny);

double* matrixMultiplication(double *matrix_a, double *matrix_b, const unsigned ld_a, const unsigned ld_b, 
  const unsigned nx, const unsigned ny, const unsigned nz);

double* getL(double *matrix, const unsigned nx, const unsigned ld, const unsigned ny);

double* getU(double *matrix, const unsigned nx, const unsigned ld, const unsigned ny);

double* multiplyLU(double *matrix, const unsigned nx, const unsigned ld, const unsigned ny);

double L2Error (double *matrix, double* target, const unsigned nx, const unsigned ld, const unsigned ny);

void printMatrix (double *matrix, const unsigned nx, const unsigned ld, const unsigned ny);

Matrix testMatrix_M(const unsigned nx, const unsigned ny);

Matrix identityMatrix_M(const unsigned nx, const unsigned ny);

Matrix matrixMultiplication(Matrix a, Matrix b);

Matrix getL(Matrix m);

Matrix getU(Matrix m);

Matrix multiplyLU(Matrix m);

double L2Error (Matrix m, Matrix t);

void printMatrix (Matrix a);

#endif