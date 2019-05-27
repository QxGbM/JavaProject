
#ifndef _DEV_TEMP_CUH
#define _DEV_TEMP_CUH

#include <pspl.cuh>

class dev_temp
{
private:
  int size;
  int length;

  int * sizes;

public:
  __host__ dev_temp (const int size_in = _DEFAULT_PTRS_LENGTH)
  {
    size = size_in > 0 ? size_in : 1;
    length = 0;
    sizes = new int [size]; 
    memset (sizes, 0, size * sizeof(int));
  }

  __host__ ~dev_temp ()
  {
    delete[] sizes;
  }

};


#endif