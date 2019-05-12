#ifndef _DEV_HIERARCHICAL_INDEX_CUH
#define _DEV_HIERARCHICAL_INDEX_CUH

#include <pspl.cuh>

class h_index 
{
private:

  int index_lvls;
  int * indexs;

  element_t type;
  int nx;
  int ny;
  int ld;
  int offset_x;
  int offset_y;

  int n_ptrs;
  void ** data_ptrs;

  const void * struct_ptr;
  const void * root_ptr;

public:

  __host__ h_index ()
  {
    index_lvls = 0;
    indexs = nullptr;
    type = empty;
    nx = ny = ld = 0;
    offset_x = offset_y = 0;
    n_ptrs = 0;
    data_ptrs = nullptr;
    struct_ptr = nullptr;
    root_ptr = nullptr;
  }

  __host__ ~h_index ()
  {
    if (index_lvls > 0) { delete[] indexs; }
    if (n_ptrs > 0) { delete[] data_ptrs; }
  }

  template <class T> __host__ h_index (const dev_hierarchical <T> * h)
  {
    index_lvls = 0;
    indexs = nullptr;

    type = hierarchical;
    nx = h -> getNx();
    ny = h -> getNy();
    ld = 0;
    offset_x = offset_y = 0;

    n_ptrs = 0;
    data_ptrs = nullptr;
    struct_ptr = h;
    root_ptr = h;
  }

  template <class T> __host__ h_index (const dev_hierarchical <T> * h, const h_index * index, const int y, const int x)
  {
    index_lvls = index -> index_lvls + 1;

    indexs = new int [index_lvls];

    for (int i = 0; i < index -> index_lvls; i++)
    { indexs[i] = (index -> indexs)[i]; }
    
    indexs[index_lvls - 1] = y * (h -> getX()) + x;

    dev_h_element <T> * element = h -> getBlock(x, y);
    type = element -> getType();
    nx = element -> getNx();
    ny = element -> getNy();
    ld = element -> getLd();

    offset_x = offset_y = 0;

    if (type == hierarchical)
    {
      n_ptrs = 0;
      data_ptrs = nullptr;
      struct_ptr = element -> getElementHierarchical();
    }
    else if (type == low_rank)
    {
      n_ptrs = 2;
      dev_low_rank <T> * lr = element -> getElementLowRank();
      data_ptrs = new void *[2] { lr -> getElements(), lr -> getElements(lr -> getOffset_VT()) };
      struct_ptr = lr;
    }
    else if (type == dense)
    {
      n_ptrs = 1;
      dev_dense <T> * d = element -> getElementDense();
      data_ptrs = new void *[1]{ d -> getElements() };
      struct_ptr = d;
    }
    else
    {
      n_ptrs = 0;
      data_ptrs = nullptr;
      struct_ptr = nullptr;
    }

    root_ptr = index -> root_ptr;

  }

  __host__ h_index (const h_index * index, const int y_start, const int x_start, const int ny_block, const int nx_block)
  {
    index_lvls = index -> index_lvls;

    indexs = new int [index_lvls];

    for (int i = 0; i < index_lvls; i++)
    { indexs[i] = (index -> indexs)[i]; }
    
    type = index -> type;
    nx = nx_block;
    ny = ny_block;
    ld = index -> ld;

    offset_x = index -> offset_x + x_start;
    offset_y = index -> offset_y + y_start;

    n_ptrs = index -> n_ptrs;
    data_ptrs = (n_ptrs > 0) ? new void * [n_ptrs] : nullptr;

    for (int i = 0; i < n_ptrs; i++)
    { data_ptrs[i] = (index -> data_ptrs)[i]; }

    struct_ptr = index -> struct_ptr;
    root_ptr = index -> root_ptr;
  }

  __host__ inline int getNx() const
  { return nx; }

  __host__ inline int getNy() const
  { return ny; }

  __host__ relation_t compare (const h_index * index) const
  {
    if (this == nullptr || index == nullptr || root_ptr != index -> root_ptr) 
    { return diff_mat; }

    for (int i = 0; i < index_lvls && i < (index -> index_lvls); i++) 
    { 
      if (indexs[i] != (index -> indexs)[i]) 
      { return same_mat_diff_branch; } 
    }

    if (index -> index_lvls != index_lvls)
    {
      printf("-- Some Intermediate node are being compared. This should not happen. --\n");
      return same_branch_diff_node; 
    }
    else
    {       
      if (offset_x == index -> offset_x && offset_y == index -> offset_y) 
      { return same_index; }
      else
      {
        const bool row_split = (index -> offset_y - offset_y >= ny) || (offset_y - index -> offset_y >= index -> ny);
        const bool col_split = (index -> offset_x - offset_x >= nx) || (offset_x - index -> offset_x >= index -> nx);

        return (row_split || col_split) ? same_node_no_overlap : same_node_overlapped;
      } 
    }
  }

  __host__ h_index * clone (h_index * addr = nullptr) const
  {
    if (this == nullptr)
    { return nullptr; }
    else if (addr == nullptr)
    { h_index * id = new h_index(); return clone(id); }
    else
    {
      addr -> index_lvls = index_lvls;
      addr -> type = type;
      addr -> nx = nx;
      addr -> ny = ny;
      addr -> ld = ld;
      addr -> offset_x = offset_x;
      addr -> offset_y = offset_y;
      addr -> n_ptrs = n_ptrs;
      addr -> struct_ptr = struct_ptr;
      addr -> root_ptr = root_ptr;

      addr -> indexs = (index_lvls > 0) ? new int [index_lvls] : nullptr;
      for (int i = 0; i < index_lvls; i++)
      { (addr -> indexs)[i] = indexs[i]; }

      addr -> data_ptrs = (n_ptrs > 0) ? new void * [n_ptrs] : nullptr;
      for (int i = 0; i < n_ptrs; i++)
      { (addr -> data_ptrs)[i] = data_ptrs[i]; }

      return addr;
    }
  }

  __host__ void print() const
  {
    printf("[%d ", index_lvls);
    for (int i = 0; i < index_lvls; i++)
    { printf("%d", indexs[i]); }
    switch (type)
    {
    case empty: printf(" E "); break;
    case dense: printf(" D "); break;
    case low_rank: printf(" LR "); break;
    case hierarchical: printf(" H "); break;
    }
    printf("(%d %d) (%d x %d b %d)] ", offset_y, offset_x, ny, nx, ld);
  }

};


#endif