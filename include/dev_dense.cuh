
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
  __host__ dev_dense (const int x, const int y, const int d = 0, const bool alloc_pivot = false)
  {
    nx = x;
    ny = y;
    ld = (x > d) ? x : d;

    cudaMallocManaged (&elements, ld * ny * sizeof(T), cudaMemAttachGlobal);
    cudaMemset (elements, 0, ld * ny * sizeof(T));
    
    pivoted = alloc_pivot;
    if (pivoted)
    {
      cudaMallocManaged (&pivot, ny * sizeof(int), cudaMemAttachGlobal);
      cudaMemset (pivot, 0, ny * sizeof(int));
    }
    else
    { pivot = nullptr; }

  }

  __host__ dev_dense (const int x, const int y, const T * A, const int ld_a, const int d = 0, const bool alloc_pivot = false)
  {
    nx = x;
    ny = y;
    ld = (x > d) ? x : d;

    cudaMallocManaged (&elements, ld * ny * sizeof(T), cudaMemAttachGlobal);
    loadArray (A, x, y, ld_a);

    pivoted = alloc_pivot;
    if (pivoted)
    {
      cudaMallocManaged (&pivot, ny * sizeof(int), cudaMemAttachGlobal);
      cudaMemset (pivot, 0, ny * sizeof(int));
    }
    else
    { pivot = nullptr; }
  }

  __host__ ~dev_dense ()
  {
    cudaFree(elements);
    cudaFree(pivot);
  }

  __host__ void loadArray (const T * A, const int nx_a, const int ny_a, const int ld_a, const int x_start = 0, const int y_start = 0)
  {
    for (int y = y_start, y_a = 0; y < ny && y_a < ny_a; y++, y_a++)
    {
      for (int x = x_start, x_a = 0; x < nx && x_a < nx_a; x++, x_a++)
      {
        elements[y * ld + x] = A[y_a * ld_a + x_a];
      }
    }
  }

  __host__ int getNx () const
  {
    return nx;
  }

  __host__ int getNy () const
  {
    return ny;
  }

  __host__ int getLd () const
  {
    return ld;
  }

  __host__ T * getElements (const int offset = 0) const
  {
    return &elements[offset];
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
        printf("%5.4f, ", e);
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
        elements[y * ld + x] = (T) (1.0 / ((x > y) ? x - y + 1 : y - x + 1));
      }
    }
  }

  __host__ void loadIdentityMatrix()
  {
    for (int x = 0; x < nx; x++)
    {
      for (int y = 0; y < ny; y++)
      {
        elements[y * ld + x] = (T)((x == y) ? 1 : 0);
      }
    }
  }

  __host__ void loadRandomMatrix(const double min, const double max, const int seed = 0)
  {
    if (seed > 0) 
    { srand(seed); }
    for(int x = 0; x < nx; x++)
    {
      for(int y = 0; y < ny; y++)
      { 
        elements[y * ld + x] = (T) (min + ((T) rand() / RAND_MAX) * (max - min)); 
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