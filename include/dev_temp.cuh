
#pragma once
#ifndef _DEV_TEMP_CUH
#define _DEV_TEMP_CUH

#include <definitions.cuh>

class dev_temp
{
private:
  int size;
  int length;

  int * sizes;

public:
  dev_temp (const int size_in = _PTRS_LENGTH)
  {
    size = size_in > 0 ? size_in : 1;
    length = 0;

    sizes = new int [size];

    memset (sizes, 0, size * sizeof(int));
  }

  ~dev_temp ()
  {
    delete[] sizes;
  }

  void resize (const int size_in)
  {
    if (size_in > 0 && size != size_in)
    {
      int * sizes_new = new int [size_in], n = size_in > size ? size : size_in;
      void ** ptrs_new = new void * [size_in];

      for (int i = 0; i < n; i++)
      { sizes_new[i] = sizes[i]; }

      for (int i = n; i < size_in; i++)
      { sizes_new[i] = 0; }

      delete[] sizes;

      sizes = sizes_new;

      size = size_in;
      length = length > size ? size : length;
    }
  }

  int requestTemp (const int tmp_size)
  {
    if (length == size)
    { resize(size * 2); }

    const int block_id = length;
    sizes[block_id] = tmp_size;

    length = length + 1;
    return block_id;

  }

  int requestTemp_2x (const int tmp_size1, const int tmp_size2)
  {
    int block_id = requestTemp(tmp_size1);
    requestTemp(tmp_size2);
    return block_id;
  }

  inline int getLength () const
  { return length; }

  real_t ** allocate () const
  {
    real_t** ptrs = new real_t * [length];
    int * offsets = new int [length], accum = 0;

    for (int i = 0; i < length; i++)
    { offsets[i] = accum; accum += sizes[i]; }

    printf("TMP length: %d.\n", accum);

    cudaMalloc(&(ptrs[0]), (size_t) accum * real_bits);
    cudaMemset(ptrs[0], 0, (size_t) accum * real_bits);

#pragma omp parallel for
    for (int i = 1; i < length; i++)
    { const int offset = offsets[i]; ptrs[i] = &(ptrs[0])[offset]; }

    delete[] offsets;
    return ptrs;
  }

  void print() const
  {
    printf("Temp Manager: Size %d. \n", size);
    for (int i = 0; i < length; i++)
    { printf("Block %d: size %d. \n", i, sizes[i]); }
    printf("\n");
  }

};


#endif