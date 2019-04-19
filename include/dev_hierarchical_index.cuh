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

  __host__ int getLevels() const { return levels; }

  __host__ int getIndex(const int level) const { return ns[level]; }

  __host__ int getOffset() const { return offset; }

  __host__ relation_t compare (const int nx0, const int ny0, const int ld0, const bool t0, 
    const h_index *in, const int nx1, const int ny1, const int ld1, const bool t1) const
  {
    if (this == nullptr || in == nullptr || matrix != in -> matrix) { return diff_matrix; }

    int n = ((in -> levels) > levels) ? levels : (in -> levels);
    for (int i = 0; i < n; i++) 
    { if (ns[i] != (in -> ns)[i]) return no_relation; }

    if (in -> levels == levels)
    {
      if (offset == in -> offset) return same_index;
      else
      {
        const int offset0 = offset, offset1 = in -> offset;

        const int row0 = (offset0 == 0) ? 0 : offset0 / ld0, col0 = offset0 - row0 * ld0;
        const int row1 = (offset1 == 0) ? 0 : offset1 / ld1, col1 = offset1 - row1 * ld1;
        const int row_diff = row1 - row0, col_diff = col1 - col0;

        const int ny0_ = t0 ? nx0 : ny0, nx0_ = t0 ? ny0 : nx0;
        const int ny1_ = t1 ? nx1 : ny1, nx1_ = t1 ? ny1 : nx1;

        const bool row_over = (row_diff >= 0 && row_diff < ny0_) || (row_diff <= 0 && row_diff + ny1_ > 0);
        const bool col_over = (col_diff >= 0 && col_diff < nx0_) || (col_diff <= 0 && col_diff + nx1_ > 0);

        return (row_over && col_over) ? diff_offset_overlapped : diff_offset_no_overlap;
      }
    }
    else
    { return (levels > n) ? contains : contained; }
  }

  __host__ h_index * child (const int index_in = -1, const int offset_in = 0) const
  {
    if (index_in >= 0)
    {
      h_index * i = new h_index(levels + 1, ns, 0, matrix);
      (i -> ns)[levels] = index_in;
      return i;
    }
    else
    {
      h_index * i = new h_index(levels, ns, offset_in, matrix);
      return i;
    }
  }

  template <class T> __host__ inline h_index * child_UxS (const dev_low_rank <T> * lr) const
  {
    return child(-1, lr -> getOffset_UxS(offset));
  }

  template <class T> __host__ inline h_index * child_VT (const dev_low_rank <T> * lr) const
  {
    return child(-1, lr -> getOffset_VT(offset));
  }

  __host__ h_index * clone () const
  {
    if (this == nullptr) 
    { return nullptr; }
    else
    { return new h_index(levels, ns, offset, matrix); }
  }

  __host__ void cloneTo (h_index * index) const
  {
    if (index != nullptr && this != nullptr)
    {
      index -> levels = levels;

      index -> ns = new int [levels];
      for (int i = 0; i < levels; i++) 
      { (index -> ns)[i] = ns[i]; }

      index -> offset = offset;
      index -> matrix = matrix;
    }
  }

  __host__ void print() const
  {
    printf("-- ");
    if (levels == 0) printf("root, ");
    for (int i = 0; i < levels; i++)
    { printf("level %d: %d, ", i, ns[i]); }
    printf("offset %d. --\n", offset);
  }

  __host__ void printShort() const
  {
    printf("[%d", levels);
    for (int i = 0; i < levels; i++)
    { printf("%d", ns[i]); }
    printf(" (%d)]", offset);
  }

};


#endif