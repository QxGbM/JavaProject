
#ifndef _DEV_TEMP_CUH
#define _DEV_TEMP_CUH

#include <pspl.cuh>

class dev_temp
{
private:
  int size;
  int length;

  int * ids;
  int * sizes;

public:
  __host__ dev_temp (const int size_in = _DEFAULT_PTRS_LENGTH)
  {
    size = size_in > 0 ? size_in : 1;
    length = 0;

    ids = new int [size];
    sizes = new int [size];

    memset (ids, 0, size * sizeof(int));
    memset (sizes, 0, size * sizeof(int));
  }

  __host__ ~dev_temp ()
  {
    delete[] ids;
    delete[] sizes;
  }

  __host__ void resize (const int size_in)
  {
    if (size_in > 0 && size != size_in)
    {
      int * ids_new = new int [size_in], * sizes_new = new int [size_in], n = size_in > size ? size : size_in;
      bool * in_use_new = new bool [size_in];
      void ** ptrs_new = new void * [size_in];

      for (int i = 0; i < n; i++)
      {
        ids_new[i] = ids[i];
        sizes_new[i] = sizes[i];
      }

      for (int i = n; i < size_in; i++)
      {
        ids_new[i] = 0;
        sizes_new[i] = 0;
      }

      delete[] ids;
      delete[] sizes;

      ids = ids_new;
      sizes = sizes_new;

      size = size_in;
      length = length > size ? size : length;
    }
  }

  __host__ int requestTemp (const int tmp_size)
  {
    int insert_pos = length;

    for (int i = 0; i < length; i++)
    {
      const int block_id = ids[i];

      if (sizes[block_id] >= tmp_size)
      { insert_pos = i; break; }
    }

    if (length == size)
    { resize(size * 2); }
    const int block_id = length;
    sizes[block_id] = tmp_size;

    for (int i = length; i > insert_pos; i--)
    { ids[i] = ids[i - 1]; }

    ids[insert_pos] = block_id;
    length = length + 1;
    return block_id;

  }

  __host__ inline int getLength () const
  { return length; }

  template <class T> __host__ T ** allocate () const
  {
    T ** ptrs = new T * [length];
    int * offsets = new int [length], accum = 0;

    for (int i = 0; i < length; i++)
    { offsets[i] = accum; accum += sizes[i]; }

    printf("TMP length: %d.\n", accum);

    cudaMalloc(&(ptrs[0]), accum * sizeof(T));
    cudaMemset(ptrs[0], 0, accum * sizeof(T));

#pragma omp parallel for
    for (int i = 1; i < length; i++)
    { const int offset = offsets[i]; ptrs[i] = &(ptrs[0])[offset]; }

    delete[] offsets;

    return ptrs;
  }

  __host__ void print() const
  {
    printf("Temp Manager: Size %d. \n", size);
    for (int i = 0; i < length; i++)
    {
      const int block_id = ids[i];
      printf("Block %d: size %d. \n", block_id, sizes[block_id]);
    }
    printf("\n");
  }

};


#endif