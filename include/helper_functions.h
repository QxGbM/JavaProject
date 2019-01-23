#ifndef _HELPER_FUNCS_H
#define _HELPER_FUNCS_H

#include <stdio.h>
#include <math.h>
#include <malloc.h>
#include <memory.h>

#define max_(a, b) (a > b) ? a : b
#define min_(a, b) (a < b) ? a : b

typedef struct {
  double *head;
  const unsigned nx;
  const unsigned ld;
  const unsigned ny;
} Matrix;

double* testMatrix(const unsigned nx, const unsigned ny)
{
  double *matrix = (double*) malloc(nx * ny * sizeof(double));
  for(unsigned x = 0; x < nx; x++)
  {
    for(unsigned y = 0; y < ny; y++)
    {
      double d = (x > y) ? x - y : y - x;
      matrix[y * nx + x] = 1.0 / (1.0 + d);
    }
  }
  return matrix;
}

double* identityMatrix(const unsigned nx, const unsigned ny)
{
  double *matrix = (double*) malloc(nx * ny * sizeof(double));
  for(unsigned x = 0; x < nx; x++)
  {
    for(unsigned y = 0; y < ny; y++)
    {
      matrix[y * nx + x] = (double) (x == y);
    }
  }
  return matrix;
}

double* randomMatrix(const unsigned nx, const unsigned ny, const double min, const double max)
{
  double *matrix = (double*) malloc(nx * ny * sizeof(double));
  for(unsigned x = 0; x < nx; x++)
  {
    for(unsigned y = 0; y < ny; y++)
    {
      double d = (double) rand() / RAND_MAX;
      matrix[y * nx + x] = min + d * (max - min);
    }
  }
  return matrix;
}

double* zeroMatrix(const unsigned nx, const unsigned ny)
{
  double *matrix = (double*) malloc(nx * ny * sizeof(double));
  memset(matrix, 0, nx * ny * sizeof(double));
  return matrix;
}

double* matrixMultiplication(double *matrix_a, double *matrix_b, const unsigned ld_a, const unsigned ld_b, const unsigned nx, const unsigned ny, const unsigned nz)
{
  double *matrix = (double*) malloc(nx * nz * sizeof(double));
  for(unsigned x = 0; x < nx; x++)
  {
    for(unsigned z = 0; z < nz; z++)
    {
      matrix[x * nz + z] = 0;
      for(unsigned y = 0; y < ny; y++)
      {
        matrix[x * nz + z] += matrix_a[x * ld_a + y] * matrix_b[y * ld_b + z];
      }
    }
  }
  return matrix;
}

double* getL(double *matrix, const unsigned nx, const unsigned ld, const unsigned ny)
{
  double *l = (double*)malloc(ny * ny * sizeof(double));
  for (unsigned i = 0; i < ny; i++) 
  {
    for (unsigned j = 0; j < ny; j++)
    {
      if (i > j && j < nx)
      { l[i * ny + j] = matrix[i * ld + j]; }
      else if (i == j)
      { l[i * ny + j] = 1.0; }
      else
      { l[i * ny + j] = 0.0; }

      if (isnan(l[i * ny + j])) { printf("getL: nan occurred on (%d, %d) matrix = %f.\n", i, j, matrix[i * ld + j]); }
    }
  }
  return l;
}

double* getU(double *matrix, const unsigned nx, const unsigned ld, const unsigned ny)
{
  double *u = (double*)malloc(nx * ny * sizeof(double));
  for (unsigned i = 0; i < ny; i++) 
  {
    for (unsigned j = 0; j < nx; j++) 
    {
      if (i <= j) { u[i * nx + j] = matrix[i * ld + j]; }
      else { u[i * nx + j] = 0; }

      if (isnan(u[i * nx + j])) { printf("getU: nan occurred on (%d, %d) matrix = %f.\n", i, j, matrix[i * ld + j]); }
    }
  }
  return u;
}

double* multiplyLU(double *matrix, const unsigned nx, const unsigned ld, const unsigned ny)
{
  double *l = getL(matrix, nx, ld, ny);
  double *u = getU(matrix, nx, ld, ny);

  return matrixMultiplication(l, u, ny, nx, ny, ny, nx);
}

double L2Error (double *matrix, double* target, const unsigned nx, const unsigned ld, const unsigned ny) 
{
  double norm = 0.0;
  for(unsigned x = 0; x < nx; x++)
  {
    for(unsigned y = 0; y < ny; y++)
    {
      double t = 0;
      if (target != nullptr) { t = matrix[y * ld + x] - target[y * ld + x]; }
      else { t = matrix[y * ld + x]; }
      norm += t * t;
    }
  }
  if (target != nullptr) { return sqrt(norm / L2Error(matrix, nullptr, nx, ld, ny)); }
  else { return norm; }
}

void printMatrix (double *matrix, const unsigned nx, const unsigned ld, const unsigned ny)
{
  for(unsigned y = 0; y < ny; y++)
  {
    for(unsigned x = 0; x < nx; x++)
    {
      printf("%5.3f, ", matrix[y * ld + x]);
    }
    printf("\n");
  }
  printf("\n");
}


Matrix testMatrix_M(const unsigned nx, const unsigned ny)
{
  return {testMatrix(nx, ny), nx, nx, ny};
}

Matrix identityMatrix_M(const unsigned nx, const unsigned ny)
{
  return {identityMatrix(nx, ny), nx, nx, ny};
}

Matrix matrixMultiplication(Matrix a, Matrix b)
{
  double *matrix_a = a.head, *matrix_b = b.head;
  const unsigned nx = a.ny, ld_a = a.ld, ny = (a.nx == b.ny) ? a.nx : 0, nz = b.nx, ld_b = b.ld;
  if (ny == 0) { printf("Matrices have unmatched dimension.\n"); return {nullptr, 0, 0, 0}; }
  else { return {matrixMultiplication(matrix_a, matrix_b, ld_a, ld_b, nx, ny, nz), nx, nx, nz}; }
}

Matrix getL(Matrix m)
{
  double *matrix = m.head;
  const unsigned nx = m.nx, ld = m.ld, ny = m.ny;
  return {getL(matrix, nx, ld, ny), nx, nx, ny};
}

Matrix getU(Matrix m)
{
  double *matrix = m.head;
  const unsigned nx = m.nx, ld = m.ld, ny = m.ny;
  return {getU(matrix, nx, ld, ny), nx, nx, ny};
}

Matrix multiplyLU(Matrix m)
{
  double *matrix = m.head;
  const unsigned nx = m.nx, ld = m.ld, ny = m.ny;
  return {multiplyLU(matrix, nx, ld, ny), nx, nx, ny};
}

double L2Error (Matrix m, Matrix t)
{
  double *matrix = m.head, *target = t.head;
  const unsigned nx = m.nx, ld = m.ld, ny = m.ny;
  return L2Error (matrix, target, nx, ld, ny);
}

void printMatrix (Matrix a)
{
  double *matrix = a.head;
  const unsigned nx = a.nx, ld = a.ld, ny = a.ny;
  printMatrix (matrix, nx, ld, ny);
}

#endif