
#ifndef _COMPRESSOR_CUH
#define _COMPRESSOR_CUH

#include <pspl.cuh>

class compressor
{

private:
  int size;
  int length;

  void ** dense_ptrs;
  int * dense_dims;

  int * ranks;
  void ** U_ptrs;
  void ** V_ptrs;

public:
  __host__ compressor (const int rnd_seed_in = 0, const int default_size = _DEFAULT_COMPRESSOR_LENGTH)
  {
    size = default_size;
    length = 0;

    const int size_3 = size * 3;

    dense_ptrs = new void * [size];
    dense_dims = new int [size_3];
    ranks = new int [size];
    U_ptrs = new void * [size];
    V_ptrs = new void * [size];

    size_t size_b = size * sizeof (void *);
    memset (dense_ptrs, 0, size_b);
    memset (U_ptrs, 0, size_b);
    memset (V_ptrs, 0, size_b);

    size_b = size * sizeof (int);
    memset (ranks, 0, size_b);

    size_b = 3 * size_b;
    memset (dense_dims, 0, size_b);

    srand(rnd_seed_in);
    double * rnd_seed = new double [_RND_SEED_LENGTH];

#pragma omp parallel for
    for (int i = 0; i < _RND_SEED_LENGTH; i++) 
    { rnd_seed[i] = (double) rand() / RAND_MAX; }

    cudaMemcpyToSymbol(dev_rnd_seed, rnd_seed, _RND_SEED_LENGTH * sizeof(double), 0, cudaMemcpyHostToDevice);
    delete[] rnd_seed;
    
  }

  __host__ ~compressor()
  {
    for (int i = 0; i < length; i++)
    { cudaFree(dense_ptrs[i]); }

    delete[] dense_ptrs;
    delete[] dense_dims;
    delete[] ranks;
    delete[] U_ptrs;
    delete[] V_ptrs;
  }

};


#endif