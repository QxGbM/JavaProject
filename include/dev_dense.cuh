
#ifndef _DEV_DENSE_CUH
#define _DEV_DENSE_CUH

#include <pspl.cuh>

template <class T> class dev_dense 
{
private:
  int nx;
  int ny;
  int ld;

  T * elements;

  bool pivoted;
  int * pivot;

public:
  __host__ dev_dense (const int x, const int y, const int d = 0, const bool alloc_pivot = true)
  {
    nx = x;
    ny = y;
    ld = (x > d) ? x : d;

    cudaMallocManaged(&elements, ld * ny * sizeof(T), cudaMemAttachGlobal);
    cudaMemset(elements, 0, ld * ny * sizeof(T));
    
    pivoted = alloc_pivot;
    if (pivoted)
    {
      cudaMallocManaged(&pivot, ny * sizeof(int), cudaMemAttachGlobal);
      cudaMemset(pivot, 0, ny * sizeof(int));
    }
    else
    {
      pivot = nullptr;
    }

  }

  __host__ ~dev_dense ()
  {
    cudaFree(elements);
    cudaFree(pivot);

    printf("-- %d x %d matrix destructed. --\n\n", ny, ld);
  }

  __host__ int * getDim3 () const
  {
    return new int[3]{ nx, ny, ld };
  }

  __host__ T * getElements () const
  {
    return elements;
  }

  __host__ int * getPivot () const
  {
    return pivot;
  }

  __host__ void print () const
  {
    printf("-- %d x %d | ld: %d --\n", ny, nx, ld);
    for (int y = 0; y < ny; y++)
    {
      for (int x = 0; x < nx; x++)
      {
        T e = elements[y * ld + x];
        if (e >= 0) { printf(" "); }
        printf("%3.4f, ", e);
      }
      printf("\n");
    }

    if (pivoted)
    {
      printf("\n-- Pivot: --\n");
      for (int y = 0; y < ny; y++)
      {
        printf("%d ", pivot[y]);
      }
      printf("\n");
    }
    
    printf("\n");
  }

  __host__ void loadTestMatrix()
  {
    for(int x = 0; x < nx; x++)
    {
      for(int y = 0; y < ny; y++)
      {
        const T d = (x > y) ? x - y : y - x;
        elements[y * ld + x] = 1.0 / (1.0 + d);
      }
    }
  }

  __host__ void loadIdentityMatrix()
  {
    int n = (nx > ny) ? ny : nx;
    for(int x = 0; x < n; x++)
    {
      elements[x * nx + x] = 1;
    }
  }

  __host__ void loadRandomMatrix(const double min, const double max, const int seed = 0)
  {
    if (seed > 0) { srand(seed); }
    for(int x = 0; x < nx; x++)
    {
      for(int y = 0; y < ny; y++)
      {
        const T d = (T) rand() / RAND_MAX;
        elements[y * ld + x] = min + d * (max - min);
      }
    }
  }

  __host__ dev_dense <T> * matrixMultiplication (const dev_dense <T> *B) const
  {
    dev_dense <T> *C = new dev_dense <T> (B -> nx, ny);
    for(int m = 0; m < ny; m++)
    {
      for(int n = 0; n < B -> nx; n++)
      {
        for(int k = 0; k < nx; k++)
        {
          (C -> elements)[m * (B -> nx) + n] += elements[m * ld + k] * (B -> elements)[k * (B -> ld) + n];
        }
      }
    }
    return C;
  }

  __host__ dev_dense <T> * restoreLU () const
  {
    dev_dense <T> *L = new dev_dense <T>(ny, ny);
    for (int i = 0; i < ny; i++)
    {
      for (int j = 0; j < ny; j++)
      {
        if (i > j && j < nx)
        {
          (L -> elements)[i * ny + j] = elements[i * ld + j];
        }
        else if (i == j)
        {
          (L -> elements)[i * ny + j] = 1;
        }
      }
    }

    dev_dense <T> *U = new dev_dense <T>(nx, ny);
    for (int i = 0; i < ny; i++)
    {
      for (int j = 0; j < nx; j++)
      {
        if (i <= j)
        {
          (U -> elements)[i * nx + j] = elements[i * ld + j];
        }
      }
    }
    
    dev_dense <T> *LU = L -> matrixMultiplication(U);
    delete L;
    delete U;
    return LU;
  }

  __host__ double L2Error (const dev_dense <T> *matrix) const
  {
    double norm = 0.0;
    for(int x = 0; x < nx; x++)
    {
      for(int y = 0; y < ny; y++)
      {
        double t = 0.0;
        if (matrix != nullptr) 
        { t = (double) (elements[y * ld + x] - (matrix -> elements)[y * (matrix -> ld) + x]); }
        else 
        { t = (double) elements[y * ld + x]; }
        norm += t * t;
      }
    }
    if (matrix != nullptr) 
    { return sqrt(norm / L2Error(nullptr)); }
    else 
    { return norm; }
  }

};


#endif