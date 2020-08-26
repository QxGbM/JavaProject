
#pragma once
#ifndef _COMPRESSOR_CUH
#define _COMPRESSOR_CUH

/*
#include <definitions.cuh>

__global__ void compressor_kernel (const int length, real_t ** __restrict__ U_ptrs, real_t** __restrict__ V_ptrs, int * __restrict__ ranks, const int * __restrict__ dims)
{
  __shared__ int shm[_SHM_SIZE];

  for (int i = block_rank(); i < length; i += grid_dim())
  {
    const int i_3 = i * 3;
    int r = blockRandomizedSVD (U_ptrs[i], V_ptrs[i], dims[i_3], dims[i_3 + 1], dims[i_3 + 2], dims[i_3 + 1], 16, 1.e-24, 64, (real_t *) shm, _SHM_SIZE * 4 / sizeof(real_t));
    if (thread_rank() == 0)
    { ranks[i] = r; }
    __syncthreads();
  }

}

class compressor
{

private:
  int size;
  int length;

  void ** U_ptrs;
  void ** V_ptrs;

  int * ranks;
  int * dims;

  void ** str_ptrs;

public:
  compressor (const int rnd_seed_in = 0, const int default_size = _COMPRESSOR_LENGTH)
  {
    size = default_size;
    length = 0;

    const int size_3 = size * 3;

    dims = new int [size_3];
    ranks = new int [size];
    U_ptrs = new void * [size];
    V_ptrs = new void * [size];
    str_ptrs = new void * [size];

    size_t size_b = size * sizeof (void *);
    memset (U_ptrs, 0, size_b);
    memset (V_ptrs, 0, size_b);
    memset (str_ptrs, 0, size_b);

    size_b = size * sizeof (int);
    memset (ranks, 0, size_b);

    size_b = 3 * size_b;
    memset (dims, 0, size_b);
    
  }

  ~compressor()
  {
    delete[] U_ptrs;
    delete[] V_ptrs;
    delete[] ranks;
    delete[] dims;
    delete[] str_ptrs;

  }

  void resize (const int size_in)
  {
    if (size_in > 0 && size_in != size)
    {
      const int size_3 = size_in * 3;
      int * dims_new = new int [size_3];
      int * ranks_new = new int [size_in];
      void ** U_ptrs_new = new void * [size_in];
      void ** V_ptrs_new = new void * [size_in];
      void ** str_ptrs_new = new void * [size_in];

      const int n = size_in > size ? size : size_in;

#pragma omp parallel for
      for (int i = 0; i < n; i++)
      {
        const int i_3 = i * 3;

        U_ptrs_new[i] = U_ptrs[i];
        V_ptrs_new[i] = V_ptrs[i];
        ranks_new[i] = ranks[i];
        dims_new[i_3] = dims[i_3];
        dims_new[i_3 + 1] = dims[i_3 + 1];
        dims_new[i_3 + 2] = dims[i_3 + 2];
        str_ptrs_new[i] = str_ptrs[i];
      }

      if (n < size_in)
      {
#pragma omp parallel for
        for (int i = size; i < size_in; i++)
        {
          const int i_3 = i * 3;

          U_ptrs_new[i] = nullptr;
          V_ptrs_new[i] = nullptr;
          dims_new[i_3] = 0;
          dims_new[i_3 + 1] = 0;
          dims_new[i_3 + 2] = 0;
          ranks_new[i] = 0;
          str_ptrs_new[i] = nullptr;
        }
      }

      delete[] U_ptrs;
      delete[] V_ptrs;
      delete[] ranks;
      delete[] dims;
      delete[] str_ptrs;

      U_ptrs = U_ptrs_new;
      V_ptrs = V_ptrs_new;
      ranks = ranks_new;
      dims = dims_new;
      str_ptrs = str_ptrs_new;

      size = size_in;
      length = (length > size_in) ? size_in : length;

    }
  }

  void compress (LowRank * M)
  {
    if (length == size)
    { resize(size * 2); }

    const int nx = M -> getNx(), ny = M -> getNy(), n = nx > ny ? ny : nx, i_3 = length * 3;
    M -> adjustRank(nx);

    dims[i_3] = nx; dims[i_3 + 1] = ny; dims[i_3 + 2] = n;

    U_ptrs[length] = M -> getUxS() -> getElements();
    V_ptrs[length] = M -> getVT() -> getElements();
    str_ptrs[length] = M;

    length++;
  }

  cudaError_t launch ()
  {
    real_t ** dev_U_ptrs, ** dev_V_ptrs;
    int * dev_ranks, * dev_dims;

    cudaMalloc(&dev_U_ptrs, length * sizeof(real_t *));
    cudaMalloc(&dev_V_ptrs, length * sizeof(real_t *));
    cudaMalloc(&dev_ranks, length * sizeof(int));
    cudaMalloc(&dev_dims, 3 * length * sizeof(int));

    cudaMemcpy(dev_U_ptrs, U_ptrs, length * sizeof(real_t *), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_V_ptrs, V_ptrs, length * sizeof(real_t *), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ranks, ranks, length * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dims, dims, 3 * length * sizeof(int), cudaMemcpyHostToDevice);

    void ** args = new void *[5] { &length, &dev_U_ptrs, &dev_V_ptrs, &dev_ranks, &dev_dims };
    cudaError_t error = cudaLaunchKernel((void *) compressor_kernel, 16, 1024, args, 0, 0);

    error = cudaDeviceSynchronize();
    fprintf(stderr, "Device: %s\n\n", cudaGetErrorString(error));

    cudaMemcpy(ranks, dev_ranks, length * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < length; i++)
    { 
      LowRank * lr = (LowRank *) str_ptrs[i];
      lr -> adjustRank(ranks[i]);
    }

    delete[] args;
    cudaFree(dev_U_ptrs);
    cudaFree(dev_V_ptrs);
    cudaFree(dev_ranks);
    cudaFree(dev_dims);

    return error;
  }

  void print()
  {
    for (int i = 0; i < length; i++)
    {
      const int i_3 = i * 3;
      printf("%d: %d x %d => %d.\n", i, dims[i_3 + 1], dims[i_3], ranks[i]);
    }
  }

};

*/
#endif