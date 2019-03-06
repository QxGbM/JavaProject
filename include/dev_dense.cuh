
#ifndef _DEV_DENSE_CUH
#define _DEV_DENSE_CUH

#include <stdio.h>
#include <math.h>
#include <cuda.h>

template <class matrixEntriesT> struct dev_dense {

  int nx;
  int ny;
  int ld;
  int dev_ld;

  matrixEntriesT *elements;
  matrixEntriesT *dev_ptr;

  bool pivoted;
  int *pivot;
  int *dev_pivot;

  __host__ dev_dense (const int x, const int y, const int d = 0, const bool alloc_pivot = true)
  {
    nx = x;
    ny = y;
    ld = (x > d) ? x : d;

    elements = new matrixEntriesT [y * ld];
    
    pivoted = alloc_pivot;
    pivot = (alloc_pivot) ? new int[y] : nullptr;
    
    for (int i = 0; i < ny * nx; i++) 
    { elements[i] = 0; if (alloc_pivot && i < ny) pivot[i] = i; }

    dev_ld = 0;
    dev_ptr = nullptr;
    dev_pivot = nullptr;
  }

  __host__ ~dev_dense ()
  {
    delete[] elements;
    if (pivoted)
    { delete[] pivot; }
    
    if (dev_ptr != nullptr) { cudaFree(dev_ptr); }
    if (dev_pivot != nullptr) { cudaFree(dev_pivot); }

    printf("-- %d x %d matrix destructed. --\n\n", ny, ld);
  }

  __host__ cudaError_t copyToDevice_Sync (const bool keep_same_ld = false)
  {
    cudaError_t error = cudaSuccess;
    dev_ld = keep_same_ld ? ld : nx;
    
    if (dev_ptr == nullptr) 
    {
      error = cudaMalloc ((void**) &dev_ptr, dev_ld * ny * sizeof(matrixEntriesT));
      if (error != cudaSuccess)
      { return error; }
      else
      { printf("-- Allocated %d x %d matrix in cuda device. --\n\n", ny, dev_ld); }
    }
  
    for (int i = 0; i < ny; i++)
    {
      error = cudaMemcpy (&dev_ptr[i * dev_ld], &elements[i * ld], nx * sizeof(matrixEntriesT), cudaMemcpyHostToDevice);
      if (error != cudaSuccess) { return error; }
    }    
    printf("-- Copied %d x %d entries from host to cuda device. --\n\n", ny, nx);

    if (pivoted)
    {
      if (dev_pivot == nullptr)
      {
        error = cudaMalloc ((void **) &dev_pivot, ny * sizeof(int));
        if (error != cudaSuccess)
        { return error; }
        else
        {
          error = cudaMemcpy (dev_pivot, pivot, ny * sizeof(int), cudaMemcpyHostToDevice);
          if (error != cudaSuccess)
          { return error; }
          else
          { printf("-- Allocated and copied row permutations for %d rows. --\n\n", ny); }
        }
      }
    }
    return cudaSuccess;
  }

  __host__ cudaError_t copyToHost_Sync (const bool free_device = false)
  {
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < ny; i++)
    {
      error = cudaMemcpy(&elements[i * ld], &dev_ptr[i * dev_ld], nx * sizeof(matrixEntriesT), cudaMemcpyDeviceToHost);
      if (error != cudaSuccess) { return error; }
    }
    
    printf("-- Copied %d x %d entries from cuda device to host. --\n\n", ny, nx);

    if (pivoted)
    {
      error = cudaMemcpy (pivot, dev_pivot, ny * sizeof(int), cudaMemcpyDeviceToHost);
      if (error != cudaSuccess)
      { return error; }
      else
      { printf("-- Copied %d row permutations from cuda device to host. --\n\n", ny); }
    }
  
    if (free_device)
    {
      error = cudaFree(dev_ptr);
      if (dev_pivot != nullptr) { error = cudaFree(dev_pivot); }
      if (error != cudaSuccess) { return error; }
      dev_ld = 0;
      dev_ptr = nullptr;
      dev_pivot = nullptr;
  
      printf("-- Freed %d x %d matrix in cuda device. --\n\n", ny, dev_ld);
    }
  
    return cudaSuccess;
  }

  /* Host Functions */

  __host__ void print () const
  {
    printf("-- %d x %d | leading dimension: %d --\n", ny, nx, ld);
    for (int y = 0; y < ny; y++)
    {
      for (int x = 0; x < nx; x++)
      {
        matrixEntriesT e = elements[y * ld + x];
        if (e >= 0) { printf(" "); }
        printf("%5.3f, ", e);
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
        matrixEntriesT d = (x > y) ? x - y : y - x;
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
        matrixEntriesT d = (matrixEntriesT) rand() / RAND_MAX;
        elements[y * nx + x] = min + d * (max - min);
      }
    }
  }

  __host__ struct dev_dense <matrixEntriesT> * matrixMultiplication (const struct dev_dense <matrixEntriesT> *B) const
  {
    struct dev_dense <matrixEntriesT> *C = new dev_dense <matrixEntriesT> (B -> nx, ny);
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

  __host__ struct dev_dense <matrixEntriesT> * getLowerTriangular () const
  {
    struct dev_dense <matrixEntriesT> *L = new dev_dense <matrixEntriesT> (ny, ny);
    for (int i = 0; i < ny; i++) 
    {
      for (int j = 0; j < ny; j++)
      {
        if (i > j && j < nx)
        { (L -> elements)[i * ny + j] = elements[i * ld + j]; }
        else if (i == j)
        { (L -> elements)[i * ny + j] = 1; }
      }
    }
    return L;
  }

  __host__ struct dev_dense <matrixEntriesT> * getUpperTriangular () const
  {
    struct dev_dense <matrixEntriesT> *U = new dev_dense <matrixEntriesT> (nx, ny);
    for (int i = 0; i < ny; i++) 
    {
      for (int j = 0; j < nx; j++) 
      {
        if (i <= j) 
        { (U -> elements)[i * nx + j] = elements[i * ld + j]; }
      }
    }
    return U;
  }

  __host__ struct dev_dense <matrixEntriesT> * restoreLU () const
  {
    struct dev_dense <matrixEntriesT> *L = getLowerTriangular(), *U = getUpperTriangular();
    struct dev_dense <matrixEntriesT> *LU = L -> matrixMultiplication(U);
    delete L;
    delete U;
    return LU;
  }

  __host__ double L2Error (const struct dev_dense <matrixEntriesT> *matrix) const
  {
    double norm = 0.0;
    for(int x = 0; x < nx; x++)
    {
      for(int y = 0; y < ny; y++)
      {
        double t = 0.0;
        if (matrix != nullptr) 
        { t = (double) (elements[y * ld + x] - (matrix -> elements)[y * ld + x]); }
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