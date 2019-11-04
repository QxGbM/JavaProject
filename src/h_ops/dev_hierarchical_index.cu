

#include <definitions.cuh>
#include <h_ops/dev_hierarchical_index.cuh>
#include <matrix/dev_hierarchical.cuh>
#include <matrix/dev_hierarchical_element.cuh>
#include <matrix/dev_low_rank.cuh>
#include <matrix/dev_dense.cuh>

h_index::h_index()
{
  index_lvls = 0;
  indexs = nullptr;
  type = empty;
  nx = ny = ld_x = ld_y = 0;
  offset_x = offset_y = rank = 0;
  n_ptrs = 0;
  data_ptrs = nullptr;
  tmp_id = -1;
  root_ptr = nullptr;
}

h_index::h_index (const h_index * index)
{
  index_lvls = index -> index_lvls;
  type = index -> type;
  nx = index -> nx;
  ny = index -> ny;
  ld_x = index -> ld_x;
  ld_y = index -> ld_y;
  offset_x = index -> offset_x;
  offset_y = index -> offset_y;
  rank = index -> rank;

  n_ptrs = index -> n_ptrs;
  tmp_id = index -> tmp_id;
  root_ptr = index -> root_ptr;

  indexs = (index_lvls > 0) ? new int [index_lvls] : nullptr;
  for (int i = 0; i < index_lvls; i++)
  { indexs[i] = (index -> indexs)[i]; }

  data_ptrs = (n_ptrs > 0) ? new void * [n_ptrs] : nullptr;
  for (int i = 0; i < n_ptrs; i++)
  { data_ptrs[i] = (index -> data_ptrs)[i]; }
}

h_index::h_index (const dev_hierarchical * h)
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
  root_ptr = h;
}

h_index::h_index (const dev_hierarchical * h, const h_index * index, const int y, const int x)
{
  index_lvls = index -> index_lvls + 1;

  indexs = new int [index_lvls];

  for (int i = 0; i < index -> index_lvls; i++)
  { indexs[i] = (index -> indexs)[i]; }
    
  indexs[index_lvls - 1] = y * (h -> getNx_blocks()) + x;

  dev_h_element * element = h -> getElement_blocks(y, x);
  type = element -> getType();
  nx = element -> getNx();
  ny = element -> getNy();

  offset_x = offset_y = 0;

  if (type == hierarchical)
  {
    ld_x = ld_y = rank = 0;
    n_ptrs = 0;
    data_ptrs = nullptr;
  }
  else if (type == low_rank)
  {
    n_ptrs = 2;
    dev_low_rank * lr = element -> getElementLowRank();
    ld_x = lr -> getUxS() -> getLd();
    ld_y = lr -> getVT() -> getLd();
    rank = lr -> getRank();
    data_ptrs = new void *[2] { lr -> getUxS() -> getElements(), lr -> getVT() -> getElements() };
  }
  else if (type == dense)
  {
    n_ptrs = 3;
    dev_dense * d = element -> getElementDense();
    ld_x = d -> getLd();
    ld_y = 0;
    rank = d -> getShadowRank();
    data_ptrs = new void *[3] { d -> getElements(), d -> getShadow_U(), d -> getShadow_VT() };
  }
  else
  {
    ld_x = ld_y = rank = 0;
    n_ptrs = 0;
    data_ptrs = nullptr;
  }

  tmp_id = -1;
  root_ptr = index -> root_ptr;

}

h_index::h_index (const h_index * index, const int y_start, const int x_start, const int ny_block, const int nx_block)
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

  root_ptr = index -> root_ptr;
}

h_index::h_index (const dev_dense * d)
{
  index_lvls = 0;
  indexs = nullptr;
  type = dense;
  nx = d -> getNx();
  ny = d -> getNy();
  ld_x = d -> getLd();
  ld_y = 0;
  offset_x = offset_y = rank = 0;
  n_ptrs = 3;
  data_ptrs = new void * [1];
  data_ptrs[0] = d -> getElements();
  data_ptrs[1] = d -> getShadow_U();
  data_ptrs[2] = d -> getShadow_VT();
  tmp_id = -1;
  root_ptr = d;
}

h_index::h_index (const dev_low_rank * lr)
{
  index_lvls = 0;
  indexs = nullptr;
  type = low_rank;
  nx = lr -> getNx();
  ny = lr -> getNy();
  ld_x = lr -> getVT() -> getLd();
  ld_y = lr -> getUxS() -> getLd();
  offset_x = offset_y = 0;
  rank = lr -> getRank();
  n_ptrs = 2;
  data_ptrs = new void * [2];
  data_ptrs[0] = lr -> getElements();
  data_ptrs[1] = lr -> getElements(ny * rank);
  tmp_id = -1;
  root_ptr = nullptr;
}

h_index::~h_index ()
{
  if (index_lvls > 0) { delete[] indexs; }
  if (n_ptrs > 0) { delete[] data_ptrs; }
}

int h_index::getNx() const
{ return nx; }

int h_index::getNy() const
{ return ny; }

int h_index::getSize() const
{ return nx * ny; }

int h_index::getNx (const int min) const
{ return min > nx ? nx : min; }

int h_index::getNy (const int min) const
{ return min > ny ? ny : min; }

int h_index::getLd_x() const
{ return ld_x; }

int h_index::getLd_y() const
{ return ld_y; }

int h_index::getOffset() const
{ return offset_y * ld_x + offset_x; }

int h_index::getOffset_x() const
{ return offset_x * ld_x; }

int h_index::getOffset_y() const
{ return offset_y * ld_y; }

int h_index::getRank() const
{ return rank; }

int h_index::getRank (const int min) const
{ return min > rank ? rank : min; }

int h_index::getTranspose() const
{ return (int) isVT(); }

relation_t h_index::compare (const h_index * index) const
{
  if (this == nullptr || index == nullptr || root_ptr != index -> root_ptr) 
  { return diff_mat; }

  for (int i = 0; i < index_lvls && i < (index -> index_lvls); i++) 
  { 
    if (indexs[i] != (index -> indexs)[i]) 
    { return same_mat_diff_branch; } 
  }

  if (index -> tmp_id != tmp_id)
  {
    return same_node_different_temp;
  }
  else if (index -> index_lvls != index_lvls)
  {
    return same_branch_diff_node; 
  }
  else if (offset_x == index -> offset_x && offset_y == index -> offset_y)
  {
    return same_index;
  }
  else
  {
    const bool all_rows = (offset_x == -1 || index -> offset_x == -1);
    const bool all_cols = (offset_y == -1 || index -> offset_y == -1);

    const bool row_split = !all_cols && ((index -> offset_y - offset_y >= ny) || (offset_y - index -> offset_y >= index -> ny));
    const bool col_split = !all_rows && ((index -> offset_x - offset_x >= nx) || (offset_x - index -> offset_x >= index -> nx));

    return (row_split || col_split) ? same_node_no_overlap : same_node_overlapped;
  }
}

h_index * h_index::clone (h_index * addr) const
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

void h_index::setU ()
{
  if (isLowRank_Full())
  { offset_x = -1; }
}

void h_index::setVT ()
{ 
  if (isLowRank_Full())
  { offset_y = -1; }
}

void h_index::setTemp_Dense (const int block_id)
{
  type = temp_dense;
  ld_x = nx;
  ld_y = 0;
  rank = 0;
  tmp_id = block_id;
  root_ptr = nullptr;

  if (data_ptrs != nullptr) 
  { delete[] data_ptrs; data_ptrs = nullptr; }

  n_ptrs = 3;
  data_ptrs = new void * [3];
  data_ptrs[0] = nullptr;
  data_ptrs[1] = nullptr;
  data_ptrs[2] = nullptr;
}

void h_index::setTemp_Low_Rank (const int block_id, const int rank_in)
{
  type = temp_low_rank;
  ld_x = rank_in;
  ld_y = rank_in;
  rank = rank_in;
  tmp_id = block_id;
  root_ptr = nullptr;

  if (data_ptrs != nullptr) 
  { delete[] data_ptrs; data_ptrs = nullptr; }

  n_ptrs = 2;
  data_ptrs = new void * [2];
  data_ptrs[0] = nullptr;
  data_ptrs[1] = nullptr;
}

void h_index::setShadow (const h_index * parent)
{
  if (parent -> isDense())
  {
    type = shadow;
    ld_x = parent -> rank;
    ld_y = parent -> rank;
    rank = parent -> rank;
    tmp_id = parent -> tmp_id;
    root_ptr = nullptr;

    if (data_ptrs != nullptr) 
    { delete[] data_ptrs; data_ptrs = nullptr; }

    n_ptrs = 2;
    data_ptrs = new void * [2];
    data_ptrs[0] = parent -> data_ptrs[1];
    data_ptrs[1] = parent -> data_ptrs[2];
  }
}

void h_index::setU_data (void * u_in, const int offset_y_in, const int ld_y_in)
{
  if (isLowRank())
  {
    if (data_ptrs[0] != nullptr)
    { printf("Warning: Overwritting U.\n"); }
    data_ptrs[0] = u_in;
    offset_y = offset_y_in;
    ld_y = ld_y_in;
  }
}

void h_index::setVT_data (void * vt_in, const int offset_x_in, const int ld_x_in)
{
  if (isLowRank())
  {
    if (data_ptrs[1] != nullptr)
    { printf("Warning: Overwritting VT.\n"); }
    data_ptrs[1] = vt_in;
    offset_x = offset_x_in;
    ld_x = ld_x_in;
  }
}

void h_index::setU_data (const h_index * index)
{
  if (index -> isLowRank())
  { setU_data(index -> data_ptrs[0], index -> offset_y, index -> ld_y); }
}

void h_index::setVT_data (const h_index * index)
{
  if (index -> isLowRank())
  { setVT_data(index -> data_ptrs[1], index -> offset_x, index -> ld_x); }
}

int h_index::getMinRank (const h_index * index, bool * a) const
{ 
  const bool b = rank < index -> rank;
  if (a != nullptr) { *a = b; }
  return b ? rank : index -> rank;
}

int h_index::getDataPointers (void ** data_ptrs_in, void ** tmp_ptrs) const
{
  int tmp_c = tmp_id, start = 0, iters = isDense() ? 1 : n_ptrs;

  for (int i = start; i < iters; i++)
  { 
    void * ptr = data_ptrs[i];
    if (ptr == nullptr && tmp_c >= 0)
    { data_ptrs_in[i] = tmp_ptrs[tmp_c]; tmp_c++; }
    else
    { data_ptrs_in[i] = ptr; }
  }

  return iters;
}

bool h_index::isDense () const
{ return (type == dense || type == temp_dense); }

bool h_index::isLowRank () const
{ return (type == low_rank || type == temp_low_rank || type == shadow); }

bool h_index::isLowRank_Full () const
{ return isLowRank() && offset_x >= 0 && offset_y >= 0; }

bool h_index::isU () const
{ return isLowRank() && offset_x == -1; }

bool h_index::isVT () const
{ return isLowRank() && offset_y == -1; }

void h_index::print() const
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

  case shadow:
  {
    if (isU())
    { printf(" S-U (%d) (%d x %d b %d)] ", offset_y, ny, rank, ld_y); }
    else if (isVT())
    { printf(" S-VT (%d) (%d x %d b %d)] ", offset_x, nx, rank, ld_x); }
    else
    { printf(" S @%d (%d %d) (%d x %d b %d, %d)] ", rank, offset_y, offset_x, ny, nx, ld_y, ld_x); }
    break; 
  }

  }
}


