#ifndef _DEV_HIERARCHICAL_INDEX_CUH
#define _DEV_HIERARCHICAL_INDEX_CUH

#include <pspl.cuh>

class multi_level_index 
{
private:

  int levels;
  int *ns;
  int offset;
  int *dim;

public:

  __host__ multi_level_index (const int levels_in = 0, const int *ns_in = nullptr, const int index_in = -1, const int offset_in = 0, const int *dim_in = nullptr)
  {
    levels = ((levels_in > 0) ? levels_in : 0) + ((index_in >= 0) ? 1 : 0);
    if (levels > 0)
    {
      ns = new int [levels];
      for (int i = 0; i < levels - 1; i++) 
      { ns[i] = (ns_in == nullptr) ? -1 : ns_in[i]; }
      ns[levels - 1] = (index_in >= 0) ? index_in : ((ns_in == nullptr) ? -1 : ns_in[levels - 1]);
    }
    else
    { ns = nullptr; }

    offset = offset_in;
    dim = new int[3]; 
    if (dim_in != nullptr) { for (int i = 0; i < 3; i++) dim[i] = dim_in[i]; }
    else { for (int i = 0; i < 3; i++) dim[i] = 1; }
  }

  __host__ ~multi_level_index ()
  {
    delete[] ns;
    delete[] dim;
  }

  __host__ void print () const
  {
    printf("-- ");
    if (levels == 0) printf("root");
    for(int i = 0; i < levels; i++)
    { printf("level %d: %d, ", i, ns[i]); }
    printf("offset %d ", offset);
    printf("dim: %d x %d by %d --\n", dim[0], dim[1], dim[2]);
  }

  __host__ void print_short () const
  {
    printf("[%d", levels);
    for(int i = 0; i < levels; i++)
    { printf("%d", ns[i]); }
    printf(" (%d) (%d,%d,%d)]", offset, dim[0], dim[1], dim[2]);
  }

  __host__ int getOffset() const
  { return offset; }

  __host__ int getNx() const
  { return dim[0]; }

  __host__ int getNy() const
  { return dim[1]; }

  __host__ int getLd() const
  { return dim[2]; }

  __host__ index_relation_t compare (const multi_level_index *in) const
  {
    if (in == nullptr) { return no_relation; }

    int n = ((in -> levels) > levels) ? levels : (in -> levels);
    for (int i = 0; i < n; i++) 
    { if (ns[i] != (in -> ns)[i]) return no_relation; }

    if (in -> levels == levels)
    {
      if (offset == in -> offset) return same_index;
      else
      {
        const int offset0 = offset, nx0 = dim[0], ny0 = dim[1], ld0 = (dim[2] == 0) ? nx0 : dim[2];
        const int offset1 = in -> offset, nx1 = (in -> dim)[0], ny1 = (in -> dim)[1], ld1 = ((in -> dim)[2] == 0) ? nx1 : (in -> dim)[2];

        if (ld0 >= nx0 && ld1 >= nx1)
        {
          const int row0 = offset0 / ld0, col0 = offset0 - row0 * ld0;
          const int row1 = offset1 / ld1, col1 = offset1 - row1 * ld1;
          const int row_diff = row1 - row0, col_diff = col1 - col0;

          const bool row_over = (row_diff >= 0 && row_diff < ny0) || (row_diff <= 0 && row_diff + ny1 > 0);
          const bool col_over = (col_diff >= 0 && col_diff < nx0) || (col_diff <= 0 && col_diff + nx1 > 0);

          return (row_over && col_over) ? diff_offset_overlapped : diff_offset_no_overlap;
        }
        else
        {
          // TODO: low rank
          return no_relation;
        }

      }
    }
    else
    { return (levels > n) ? contains : contained; }
  }

  __host__ multi_level_index * child (const int index_in = -1, const int offset_in = 0, const int *dim_in = nullptr) const
  {
    multi_level_index * i = new multi_level_index(levels, ns, index_in, offset_in, dim_in);
    return i;
  }

  __host__ multi_level_index * clone() const
  {
    multi_level_index * i = new multi_level_index(levels, ns, -1, offset, dim);
    return i;
  }

};

#endif