
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
    pivoted = alloc_pivot;

    if (cudaMallocManaged(&elements, ld * ny * sizeof(T), cudaMemAttachGlobal) != cudaSuccess)
    { 
      fprintf(stderr, "Error Allocating Dense: %s\n\n", cudaGetErrorString(cudaGetLastError()));
      elements = nullptr;
      pivot = nullptr;
    }
    else if (pivoted && cudaMallocManaged(&pivot, ny * sizeof(int), cudaMemAttachGlobal) != cudaSuccess)
    { 
      fprintf(stderr, "Error Allocating Dense Pivot: %s\n\n", cudaGetErrorString(cudaGetLastError()));
      pivoted = false;
      pivot = nullptr;
    }
    else
    {
      cudaMemset(elements, 0, ld * ny * sizeof(T));
    }
  }

  __host__ dev_dense (const int x, const int y, const T * A, const int ld_a, const int d = 0, const bool alloc_pivot = false)
  {
    nx = x;
    ny = y;
    ld = (x > d) ? x : d;
    pivoted = alloc_pivot;

    if (cudaMallocManaged(&elements, ld * ny * sizeof(T), cudaMemAttachGlobal) != cudaSuccess)
    {
      fprintf(stderr, "Error Allocating Dense: %s\n\n", cudaGetErrorString(cudaGetLastError()));
      elements = nullptr;
      pivot = nullptr;
    }
    else if (pivoted && cudaMallocManaged(&pivot, ny * sizeof(int), cudaMemAttachGlobal) != cudaSuccess)
    {
      fprintf(stderr, "Error Allocating Dense Pivot: %s\n\n", cudaGetErrorString(cudaGetLastError()));
      pivoted = false;
      pivot = nullptr;
    }
    else
    {
      loadArray(A, x, y, ld_a);
    }
  }

  __host__ ~dev_dense ()
  {
    cudaFree(elements);
    if (pivoted)
    { cudaFree(pivot); }
  }

  __host__ inline int getNx () const { return nx; }

  __host__ inline int getNy () const { return ny; }

  __host__ inline int getLd () const { return ld; }

  __host__ inline T * getElements (const int offset = 0) const { return &elements[offset]; }

  __host__ inline int * getPivot (const int offset = 0) const { return (pivoted) ? &pivot[offset / ld] : nullptr; }

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

  __host__ void resize (const int ld_in, const int ny_in)
  {
    resizeColumn (ld_in);
    resizeRow (ny_in);
  }

  __host__ void resizeColumn (const int ld_in)
  {
    if (ld_in > 0 && ld_in != ld)
    {
      T * e = nullptr;
      cudaMallocManaged (&e, ld_in * ny * sizeof(T), cudaMemAttachGlobal);
      for (int y = 0; y < ny; y++)
      {
        for (int x = 0; x < nx && x < ld_in; x++)
        { e[y * ld_in + x] = elements[y * ld + x]; }
      }
      cudaFree(elements);
      ld = ld_in;
      nx = (nx > ld) ? ld : nx;
      elements = e;
    }
  }

  __host__ void resizeRow (const int ny_in)
  {
    if (ny_in > 0 && ny_in != ny)
    {
      T * e = nullptr;
      cudaMallocManaged (&e, ld * ny_in * sizeof(T), cudaMemAttachGlobal);
      for (int y = 0; y < ny_in && y < ny; y++)
      {
        for (int x = 0; x < nx; x++)
        { e[y * ld + x] = elements[y * ld + x]; }
      }
      if (pivoted)
      {
        int * p = nullptr;
        cudaMallocManaged (&p, ny_in * sizeof(int), cudaMemAttachGlobal);
        cudaFree(pivot);
        pivot = p;
      }
      cudaFree(elements);
      ny = ny_in;
      elements = e;
    }
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

  __host__ void loadTestMatrix(const int x_start = 0, const int y_start = 0)
  {
    for(int i = 0; i < ny; i++)
    {
      const int y = y_start + i;
      for(int j = 0; j < nx; j++)
      {
        const int x = x_start + j;
        elements[i * ld + j] = (T) (1.0 / (1.0 + ((x > y) ? x - y : y - x)));
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
          (C -> elements)[m * (C -> ld) + n] += elements[m * ld + k] * (B -> elements)[k * (B -> ld) + n];
        }
      }
    }
    return C;
  }

  __host__ dev_dense <T> * transpose() const
  {
    dev_dense <T> *C = new dev_dense <T> (ny, nx);
    for (int m = 0; m < ny; m++)
    {
      for (int n = 0; n < nx; n++)
      {
        (C -> elements)[n * (C -> ld) + m] = elements[m * ld + n];
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

  __host__ double sqrSum() const
  {
    double sum = 0.0;
    for (int x = 0; x < nx; x++)
    {
      for (int y = 0; y < ny; y++)
      {
        double t = (double)elements[y * ld + x];
        sum += t * t;
      }
    }
    return sum;
  }

  __host__ double L2Error (const dev_dense <T> *matrix) const
  {
    double norm = 0.0;
    for(int x = 0; x < nx; x++)
    {
      for(int y = 0; y < ny; y++)
      {
        double t = (double) (elements[y * ld + x] - (matrix -> elements)[y * (matrix -> ld) + x]);
        norm += t * t;
      }
    }
    return sqrt(norm / sqrSum());
  }

};


#endif