
#ifndef _DEV_TEMP_CUH
#define _DEV_TEMP_CUH

#include <pspl.cuh>

class dev_temp
{
private:
  int size;
  int length;

  int * ids;
  bool * in_use;
  int * sizes;
  void ** ptrs;

public:
  __host__ dev_temp (const int size_in = _DEFAULT_PTRS_LENGTH)
  {
    size = size_in > 0 ? size_in : 1;
    length = 0;

    ids = new int [size];
    in_use = new bool [size];
    sizes = new int [size];
    ptrs = new void * [size];

    memset (ids, 0, size * sizeof(int));
    memset (in_use, 0, size * sizeof(bool));
    memset (sizes, 0, size * sizeof(int));
    memset (ptrs, 0, size * sizeof(void *));
  }

  __host__ ~dev_temp ()
  {
    for (int i = 0; i < size; i++)
    { cudaFree(ptrs[i]); }

    delete[] ids;
    delete[] in_use;
    delete[] sizes;
    delete[] ptrs;
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
        in_use_new[i] = in_use[i];
        ptrs_new[i] = ptrs[i];
      }

      for (int i = n; i < size_in; i++)
      {
        ids_new[i] = 0;
        sizes_new[i] = 0;
        in_use_new[i] = false;
        ptrs_new[i] = nullptr;
      }

      delete[] ids;
      delete[] in_use;
      delete[] sizes;
      delete[] ptrs;

      ids = ids_new;
      in_use = in_use_new;
      sizes = sizes_new;
      ptrs = ptrs_new;

      size = size_in;
      length = length > size ? size : length;
    }
  }

  __host__ int requestTemp (const int tmp_size)
  {
    int free_block_i = -1, insert_pos = 0;

    for (int i = 0; i < length; i++)
    {
      const int block_id = ids[i];
      const bool not_use = !in_use[block_id], cmp = sizes[block_id] >= tmp_size;

      if (!cmp)
      { 
        insert_pos = i + 1;
        if (not_use)
        { free_block_i = i; }
      }
      else if (not_use)
      { 
        in_use[block_id] = true;
        return block_id; 
      }
    }

    if (free_block_i != -1)
    {
      const int free_block = ids[free_block_i];
      sizes[free_block] = tmp_size;
      in_use[free_block] = true;
      for (int i = free_block_i; i < insert_pos; i++)
      { ids[i] = ids[i + 1]; }
      ids[insert_pos] = free_block;
      return free_block;
    }
    else
    {
      if (length == size)
      { resize(size * 2); }
      const int block_id = length;
      sizes[block_id] = tmp_size;
      in_use[block_id] = true;
      for (int i = length; i > insert_pos; i--)
      { ids[i] = ids[i - 1]; }
      ids[insert_pos] = block_id;
      length = length + 1;
      return block_id;
    }

  }

  __host__ inline void freeTemp (const int block_id)
  { in_use[block_id] = false; }

  __host__ void print() const
  {
    printf("Temp Manager: Size %d. \n", size);
    for (int i = 0; i < length; i++)
    {
      const int block_id = ids[i];
      printf("Block %d: in-use %d, size %d. \n", block_id, in_use[block_id], sizes[block_id]);
    }
    printf("\n");
  }

};


#endif