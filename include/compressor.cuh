
#ifndef _COMPRESSOR_CUH
#define _COMPRESSOR_CUH

#include <pspl.cuh>

template <class T, int shm_size, int rank>
__global__ void compressor_kernel (const int length, T ** __restrict__ U_ptrs, T ** __restrict__ V_ptrs, int * __restrict__ ranks, const int * __restrict__ dims)
{
  __shared__ int shm[shm_size];

  for (int i = block_rank(); i < length; i += grid_dim())
  {
    const int i_3 = i * 3;
    int r = blockRandomizedSVD <T> (U_ptrs[i], V_ptrs[i], dims[i_3], dims[i_3 + 1], dims[i_3 + 2], dims[i_3 + 1], rank, 1.e-7, 100, (T *) shm, shm_size * 4 / sizeof(T));
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
  __host__ compressor (const int rnd_seed_in = 0, const int default_size = _DEFAULT_COMPRESSOR_LENGTH)
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

    double * rnd_seed = new double [_RND_SEED_LENGTH];
    srand(rnd_seed_in);

#pragma omp parallel for
    for (int i = 0; i < _RND_SEED_LENGTH; i++) 
    { rnd_seed[i] = (double) rand() / RAND_MAX; }

    cudaMemcpyToSymbol(dev_rnd_seed, rnd_seed, _RND_SEED_LENGTH * sizeof(double), 0, cudaMemcpyHostToDevice);
    delete[] rnd_seed;
    
  }

  __host__ ~compressor()
  {
    delete[] U_ptrs;
    delete[] V_ptrs;
    delete[] ranks;
    delete[] dims;
    delete[] str_ptrs;

  }

  __host__ void resize (const int size_in)
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

  template <class T> __host__ void compress (dev_low_rank <T> * M)
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

  template <class T, int shm_size, int rank> __host__ cudaError_t launch ()
  {
    T ** dev_U_ptrs, ** dev_V_ptrs;
    int * dev_ranks, * dev_dims;

    cudaMalloc(&dev_U_ptrs, length * sizeof(T *));
    cudaMalloc(&dev_V_ptrs, length * sizeof(T *));
    cudaMalloc(&dev_ranks, length * sizeof(int));
    cudaMalloc(&dev_dims, 3 * length * sizeof(int));

    cudaMemcpy(dev_U_ptrs, U_ptrs, length * sizeof(T *), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_V_ptrs, V_ptrs, length * sizeof(T *), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ranks, ranks, length * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_dims, dims, 3 * length * sizeof(int), cudaMemcpyHostToDevice);

    void ** args = new void *[5] { &length, &dev_U_ptrs, &dev_V_ptrs, &dev_ranks, &dev_dims };
    cudaError_t error = cudaLaunchKernel((void *) compressor_kernel <T, shm_size, rank>, 16, 1024, args, 0, 0);

    error = cudaDeviceSynchronize();
    fprintf(stderr, "Device: %s\n\n", cudaGetErrorString(error));

    cudaMemcpy(ranks, dev_ranks, length * sizeof(int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < length; i++)
    { 
      printf("%d: %d\n", i, ranks[i]);
      ((dev_low_rank <T> *) str_ptrs[i]) -> adjustRank(ranks[i]);
    }

    delete[] args;
    cudaFree(dev_U_ptrs);
    cudaFree(dev_V_ptrs);
    cudaFree(dev_ranks);
    cudaFree(dev_dims);

    return error;
  }

  __host__ void print()
  {
    for (int i = 0; i < length; i++)
    {
      const int i_3 = i * 3;
      printf("%d: %d x %d.\n", i, dims[i_3 + 1], dims[i_3]);
    }
  }

};


#endif