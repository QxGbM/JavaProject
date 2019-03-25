#ifndef _DEV_HIERARCHICAL_INDEX_CUH
#define _DEV_HIERARCHICAL_INDEX_CUH

#include <pspl.cuh>

class h_index 
{
private:

  int levels;
  int * ns;
  int offset;
  const void * matrix;

public:

  __host__ h_index (const int levels_in = 0, const int *ns_in = nullptr, const int offset_in = 0, const void * matrix_in = nullptr)
  {
    levels = (levels_in >= 0) ? levels_in : 0;

    ns = new int [levels];
    for (int i = 0; i < levels; i++) 
    { ns[i] = (ns_in == nullptr) ? -1 : ns_in[i]; }

    offset = offset_in;
    matrix = matrix_in;

  }

  __host__ ~h_index ()
  {
    delete[] ns;
  }

  __host__ void print () const
  {
    printf("-- ");
    if (levels == 0) printf("root, ");
    for(int i = 0; i < levels; i++)
    { printf("level %d: %d, ", i, ns[i]); }
    printf("offset %d. --\n", offset);
  }

  __host__ void printShort () const
  {
    printf("[%d", levels);
    for(int i = 0; i < levels; i++)
    { printf("%d", ns[i]); }
    printf(" (%d)]", offset);
  }

  __host__ int getOffset() const
  { return offset; }

  __host__ relation_t compare (const h_index *in, const int * my_dim, const int * dim_in) const
  {
    if (in == nullptr) { return no_relation; }
    if (matrix != in -> matrix) { return diff_matrix; }

    int n = ((in -> levels) > levels) ? levels : (in -> levels);
    for (int i = 0; i < n; i++) 
    { if (ns[i] != (in -> ns)[i]) return no_relation; }

    if (in -> levels == levels)
    {
      if (offset == in -> offset) return same_index;
      else
      {
        const int offset0 = offset, nx0 = my_dim[0], ny0 = my_dim[1], ld0 = (my_dim[2] == 0) ? nx0 : my_dim[2];
        const int offset1 = in -> offset, nx1 = dim_in[0], ny1 = dim_in[1], ld1 = (dim_in[2] == 0) ? nx1 : dim_in[2];

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

  __host__ h_index * child (const int index_in, const int offset_in = 0) const
  {
    h_index * i = new h_index (levels + 1, ns, offset_in, matrix);
    (i -> ns)[levels] = index_in;
    return i;
  }

  __host__ h_index * clone() const
  {
    h_index * i = new h_index (levels, ns, offset, matrix);
    return i;
  }

};


#endif