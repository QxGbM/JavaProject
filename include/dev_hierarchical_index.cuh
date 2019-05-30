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
  int ld_x;
  int ld_y;
  int offset_x;
  int offset_y;
  int rank;

  int n_ptrs;
  void ** data_ptrs;
  int tmp_id;

  const void * struct_ptr;
  const void * root_ptr;

public:

  __host__ h_index ()
  {
    index_lvls = 0;
    indexs = nullptr;
    type = empty;
    nx = ny = ld_x = ld_y = 0;
    offset_x = offset_y = rank = 0;
    n_ptrs = 0;
    data_ptrs = nullptr;
    tmp_id = -1;
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
    nx = h -> getNx_abs();
    ny = h -> getNy_abs();
    ld_x = ld_y = 0;
    offset_x = offset_y = rank = 0;

    n_ptrs = 0;
    data_ptrs = nullptr;
    tmp_id = -1;
    struct_ptr = h;
    root_ptr = h;
  }

  template <class T> __host__ h_index (const dev_hierarchical <T> * h, const h_index * index, const int y, const int x)
  {
    index_lvls = index -> index_lvls + 1;

    indexs = new int [index_lvls];

    for (int i = 0; i < index -> index_lvls; i++)
    { indexs[i] = (index -> indexs)[i]; }
    
    indexs[index_lvls - 1] = y * (h -> getNx_blocks()) + x;

    dev_h_element <T> * element = h -> getElement_blocks(y, x);
    type = element -> getType();
    nx = element -> getNx();
    ny = element -> getNy();

    offset_x = offset_y = 0;
    tmp_id = -1;

    if (type == hierarchical)
    {
      ld_x = ld_y = rank = 0;
      n_ptrs = 0;
      data_ptrs = nullptr;
      struct_ptr = element -> getElementHierarchical();
    }
    else if (type == low_rank)
    {
      n_ptrs = 2;
      dev_low_rank <T> * lr = element -> getElementLowRank();
      ld_x = lr -> getUxS() -> getLd();
      ld_y = lr -> getVT() -> getLd();
      rank = lr -> getRank();
      data_ptrs = new void *[2] { lr -> getUxS() -> getElements(), lr -> getVT() -> getElements() };
      struct_ptr = lr;
    }
    else if (type == dense)
    {
      n_ptrs = 1;
      dev_dense <T> * d = element -> getElementDense();
      ld_x = d -> getLd();
      ld_y = rank = 0;
      data_ptrs = new void *[1] { d -> getElements() };
      struct_ptr = d;
    }
    else
    {
      ld_x = ld_y = rank = 0;
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
    ld_x = index -> ld_x;
    ld_y = index -> ld_y;

    offset_x = index -> offset_x + x_start;
    offset_y = index -> offset_y + y_start;
    rank = index -> rank;

    n_ptrs = index -> n_ptrs;
    data_ptrs = (n_ptrs > 0) ? new void * [n_ptrs] : nullptr;
    tmp_id = index -> tmp_id;

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
      addr -> ld_x = ld_x;
      addr -> ld_y = ld_y;
      addr -> offset_x = offset_x;
      addr -> offset_y = offset_y;
      addr -> rank = rank;
      addr -> n_ptrs = n_ptrs;
      addr -> tmp_id = tmp_id;
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

  __host__ h_index * getU (h_index * addr = nullptr) const
  { clone(addr); addr -> offset_x = -1; return addr; }

  __host__ h_index * getVT (h_index * addr = nullptr) const
  { clone(addr); addr -> offset_y = -1; return addr; }

  __host__ inline int getSize_U () const
  { return ny * rank; }

  __host__ inline int getSize_V () const
  { return nx * rank; }

  __host__ h_index * generateTemp_Dense (const int block_id, h_index * addr = nullptr) const
  {
    if (this == nullptr || type != low_rank)
    { return nullptr; }
    else if (addr == nullptr)
    { h_index * id = new h_index(); return generateTemp_Dense(block_id, id); }
    else
    {
      addr -> index_lvls = index_lvls;
      addr -> type = temp_dense;
      addr -> nx = nx;
      addr -> ny = ny;
      addr -> ld_x = ld_x;
      addr -> ld_y = 0;
      addr -> offset_x = offset_x;
      addr -> offset_y = offset_y;
      addr -> rank = 0;
      addr -> n_ptrs = 1;
      addr -> tmp_id = block_id;
      addr -> struct_ptr = nullptr;
      addr -> root_ptr = nullptr;

      addr -> indexs = (index_lvls > 0) ? new int [index_lvls] : nullptr;
      for (int i = 0; i < index_lvls; i++)
      { (addr -> indexs)[i] = indexs[i]; }

      addr -> data_ptrs = new void * [1];
      (addr -> data_ptrs)[0] = nullptr;

      return addr;
    }
  }

  __host__ h_index * generateTemp_LR_SameU (const int block_id, h_index * addr = nullptr) const
  {
    if (this == nullptr || type != low_rank)
    { return nullptr; }
    else if (addr == nullptr)
    { h_index * id = new h_index(); return generateTemp_LR_SameU(block_id, id); }
    else
    {
      addr -> index_lvls = index_lvls;
      addr -> type = temp_low_rank;
      addr -> nx = nx;
      addr -> ny = ny;
      addr -> ld_x = ld_x;
      addr -> ld_y = rank;
      addr -> offset_x = offset_x;
      addr -> offset_y = offset_y;
      addr -> rank = rank;
      addr -> n_ptrs = 2;
      addr -> tmp_id = block_id;
      addr -> struct_ptr = nullptr;
      addr -> root_ptr = nullptr;

      addr -> indexs = (index_lvls > 0) ? new int [index_lvls] : nullptr;
      for (int i = 0; i < index_lvls; i++)
      { (addr -> indexs)[i] = indexs[i]; }

      addr -> data_ptrs = new void * [2];
      (addr -> data_ptrs)[0] = data_ptrs[0];
      (addr -> data_ptrs)[1] = nullptr;

      return addr;
    }
  }

  __host__ h_index * generateTemp_LR_SameV (const int block_id, h_index * addr = nullptr) const
  {
    if (this == nullptr || type != low_rank)
    { return nullptr; }
    else if (addr == nullptr)
    { h_index * id = new h_index(); return generateTemp_LR_SameV(block_id, id); }
    else
    {
      addr -> index_lvls = index_lvls;
      addr -> type = temp_low_rank;
      addr -> nx = nx;
      addr -> ny = ny;
      addr -> ld_x = rank;
      addr -> ld_y = ld_y;
      addr -> offset_x = offset_x;
      addr -> offset_y = offset_y;
      addr -> rank = rank;
      addr -> n_ptrs = 2;
      addr -> tmp_id = block_id;
      addr -> struct_ptr = nullptr;
      addr -> root_ptr = nullptr;

      addr -> indexs = (index_lvls > 0) ? new int [index_lvls] : nullptr;
      for (int i = 0; i < index_lvls; i++)
      { (addr -> indexs)[i] = indexs[i]; }

      addr -> data_ptrs = new void * [2];
      (addr -> data_ptrs)[0] = nullptr;
      (addr -> data_ptrs)[1] = data_ptrs[1];

      return addr;
    }
  }

  __host__ inline void getDataPointers (void *** data_ptrs_in, int * n_ptrs_in) const
  { *data_ptrs_in = data_ptrs; *n_ptrs_in = n_ptrs; }

  __host__ int writeParametersTo (int * inst) const
  {
    switch (type)
    {
    case dense: case temp_dense:
      inst[0] = offset_x * ld_x + offset_y; inst[1] = ld_x; return 2;
    case low_rank: case temp_low_rank:
      inst[0] = offset_x; inst[1] = offset_y; inst[2] = ld_x; inst[3] = ld_y; inst[4] = rank; return 5;
    default:
      return 0;
    }
  }

  __host__ inline bool isU () const
  { return (type == low_rank || type == temp_low_rank) && offset_x == -1; }

  __host__ inline bool isVT () const
  { return (type == low_rank || type == temp_low_rank) && offset_y == -1; }

  __host__ void print() const
  {
    printf("[%d ", index_lvls);
    for (int i = 0; i < index_lvls; i++)
    { printf("%d", indexs[i]); }
    switch (type)
    {
    case empty: 
    { printf(" E "); break; }

    case dense: 
    { printf(" D (%d %d) (%d x %d b %d)] ", offset_y, offset_x, ny, nx, ld_x); break; }

    case low_rank:
    {
      if (isU())
      { printf(" LR-U (%d) (%d x %d b %d)] ", offset_y, ny, rank, ld_y); }
      else if (isVT())
      { printf(" LR-VT (%d) (%d x %d b %d)] ", offset_x, nx, rank, ld_x); }
      else
      { printf(" LR @%d (%d %d) (%d x %d b %d, %d)] ", rank, offset_y, offset_x, ny, nx, ld_y, ld_x); }
      break; 
    }

    case hierarchical: 
    { printf(" H (%d %d) (%d x %d)] ", offset_y, offset_x, ny, nx); break; }

    case temp_dense: 
    { printf(" T-D #%d (%d %d) (%d x %d b %d)] ", tmp_id, offset_y, offset_x, ny, nx, ld_x); break; }

    case temp_low_rank: 
    { 
      if (isU())
      { printf(" T-LR-U #%d (%d) (%d x %d b %d)] ", tmp_id, offset_y, ny, rank, ld_y); }
      else if (isVT())
      { printf(" T-LR-VT #%d (%d) (%d x %d b %d)] ", tmp_id, offset_x, nx, rank, ld_x); }
      else
      { printf(" T-LR @%d #%d (%d %d) (%d x %d b %d, %d)] ", rank, tmp_id, offset_y, offset_x, ny, nx, ld_y, ld_x); }
      break; 
    }

    }
  }

};


#endif