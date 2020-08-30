
#include <matrix/hierarchical.cuh>

Hierarchical::Hierarchical(const int m, const int n, const int part_y, const int part_x) : Element (element_t::hierarchical, 0, 0) {
  index_tree(m, n, part_y, part_x);
  elements = vector<Element*>((size_t)part_y * part_x, nullptr);
}

Hierarchical::Hierarchical(const int m, const int n, const int part_y, const int part_x, const int abs_y, const int abs_x) : Element(element_t::hierarchical, abs_y, abs_x) {
  index_tree(m, n, part_y, part_x);
  elements = vector<Element*>((size_t)part_y * part_x, nullptr);
}

void Hierarchical::index_tree(const int m, const int n, const int part_y, const int part_x) {
  row_i = vector<int>((size_t)part_y + 1, 0);
  col_i = vector<int>((size_t)part_x + 1, 0);
  int y_block = m / part_y;
  int x_block = n / part_x;
  int sum = 0;
  for (auto iter = row_i.begin(); iter != row_i.end(); iter++)
  { *iter = sum; sum += y_block; }
  row_i[part_y] = m;
  sum = 0;
  for (auto iter = col_i.begin(); iter != col_i.end(); iter++)
  { *iter = sum; sum += x_block; }
  col_i[part_x] = n;
}

Hierarchical::~Hierarchical() {
  for (auto iter = elements.begin(); iter != elements.end(); iter++) { 
    if (*iter != nullptr)
    { delete *iter; }
  }
  Element::~Element();
}

Hierarchical* Hierarchical::getElementHierarchical() 
{ return this; }

int Hierarchical::getRowDimension() const
{ return row_i[row_i.size() - 1]; }

int Hierarchical::getColumnDimension() const
{ return col_i[col_i.size() - 1]; }

int Hierarchical::getPartY() const
{ return (int)row_i.size() - 1; }

int Hierarchical::getPartX() const
{ return (int)col_i.size() - 1; }

bool Hierarchical::in_tree(const int i, const int j) const
{ return i >= 0 && j >= 0 && i < getPartY() && j < getPartX(); }

void Hierarchical::setElement(Dense* d, const int i, const int j) {
  if (in_tree(i, j)) {
    int abs_y = row_i[i];
    int abs_x = col_i[j];
    d->setLocs(abs_y, abs_x);
    elements[(size_t)j * getPartY() + i] = dynamic_cast<Element*>(d);
  }
}

void Hierarchical::setElement(LowRank* lr, const int i, const int j) {
  if (in_tree(i, j)) {
    int abs_y = row_i[i];
    int abs_x = col_i[j];
    lr->setLocs(abs_y, abs_x);
    elements[(size_t)j * getPartY() + i] = dynamic_cast<Element*>(lr);
  }
}

void Hierarchical::setElement(Hierarchical* h, const int i, const int j) {
  if (in_tree(i, j)) {
    int abs_y = row_i[i];
    int abs_x = col_i[j];
    h->setLocs(abs_y, abs_x);
    elements[(size_t)j * getPartY() + i] = dynamic_cast<Element*>(h);
  }
}

Element* Hierarchical::getChild(const int i, const int j) const
{ return in_tree(i, j) ? elements[(size_t)j * getPartY() + i] : nullptr; }

void Hierarchical::findChild (int& i, int& j, int& b_i, int& b_j) const {
  b_i = b_j = 0;
  while (b_i < getPartY() && i >= row_i[(size_t)b_i + 1]) { b_i++; }
  while (b_j < getPartX() && j >= col_i[(size_t)b_j + 1]) { b_j++; }

  if (b_i < getPartY() && b_j < getPartX())
  { i -= row_i[b_i]; j -= col_i[b_j]; }
  else
  { b_i = b_j = -1; }
}

Dense * Hierarchical::convertToDense() const {
 
  return nullptr;
}


void Hierarchical::load(ifstream& stream) {
  for (auto iter = elements.begin(); iter != elements.end(); iter++) {
    (*iter)->load(stream);
  }
}

void Hierarchical::load(const real_t* arr, const int ld) {
  for (int x = 0; x < getPartX(); x++) {
    int x_i = col_i[x] * ld;
    for (int y = 0; y < getPartY(); y++) {
      int y_i = row_i[y] + x_i;
      getChild(y, x)->load(&arr[y_i], ld);
    }
  }
}

void Hierarchical::print() const {
  print(vector<int>(), vector<int>{0, 16, 0, 16});
}

void Hierarchical::print(vector<int> &indices, vector<int>& config) const {
  int count = 0;
  for (auto iter = elements.begin(); iter != elements.end(); iter++) {
    indices.push_back(count++);
    (*iter)->print(indices, config);
    indices.pop_back();
  }
}






/*

h_ops_tree * Hierarchical::generateOps_GETRF (const h_index * self, dev_temp * tmp_mngr) const
{
  h_ops_tree * op = new h_ops_tree (getrf, self);

  int n = nx > ny ? ny : nx, * child_offset = new int[(size_t) n + 1];
  child_offset[0] = 0;

  for (int i = 1; i <= n; i++)
  { child_offset[i] = child_offset[i - 1] + (nx - i + 1) * (ny - i + 1); }

  op -> resizeChildren(child_offset[n]);

#pragma omp parallel for if (omp_in_parallel() == 0)
  for (int i = 0; i < n; i++)
  {
    const h_index index_i = h_index (this, self, i, i);
    h_ops_tree * op_i = elements[i * nx + i].generateOps_GETRF(&index_i, tmp_mngr);
    op -> setChild(op_i, child_offset[i]);
    delete op_i;
    const int rows = ny - i - 1, cols = nx - i - 1;

    for (int j = i + 1; j < nx; j++)
    {
      const h_index index_j = h_index (this, self, i, j);
      h_ops_tree * op_j = elements[i * nx + i].generateOps_TRSML(&index_i, &elements[i * nx + j], &index_j, tmp_mngr);
      op -> setChild(op_j, child_offset[i] + j - i);
      delete op_j;
    }

    for (int j = i + 1; j < ny; j++)
    {
      const h_index index_j = h_index (this, self, j, i);
      h_ops_tree * op_j = elements[i * nx + i].generateOps_TRSMR(&index_i, &elements[j * nx + i], &index_j, tmp_mngr);
      op -> setChild(op_j, child_offset[i] + cols + j - i);
      delete op_j;
    }

    for (int j = 0; j < rows * cols; j++)
    {
      const int row = j / cols + i + 1, col = j - (row - i - 1) * cols + i + 1;
      const h_index index_j = h_index (this, self, row, i), index_k = h_index (this, self, i, col), index_m = h_index (this, self, row, col);
      h_ops_tree * op_j = elements[row * nx + col].generateOps_GEMM(&index_m, &elements[row * nx + i], &index_j, &elements[i * nx + col], &index_k, tmp_mngr);
      op -> setChild(op_j, child_offset[i] + rows + cols + j + 1);
      delete op_j;
    }
  }

  delete[] child_offset;
  return op;
}

h_ops_tree * Hierarchical::generateOps_TRSML (const h_index * self, const Dense * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  h_ops_tree * op = new h_ops_tree (trsml, index_b, self);

  int n = nx > ny ? ny : nx, * child_offset = new int[(size_t) n + 1];
  child_offset[0] = 0;

  for (int i = 1; i <= n; i++)
  { child_offset[i] = child_offset[i - 1] + ny - i + 1; }

  op -> resizeChildren(child_offset[n]);

#pragma omp parallel for if (omp_in_parallel() == 0)
  for (int i = 0; i < n; i++)
  {
    h_index index_i = h_index (this, self, i, i), index_bi = h_index (index_b, y_offsets[i], 0, index_i.getNy(), index_b -> getNx());
    h_ops_tree * op_i = elements[i * nx + i].generateOps_TRSML(&index_i, B, &index_bi, tmp_mngr);
    op -> setChild(op_i, child_offset[i]);
    delete op_i;

    for (int j = i + 1; j < ny; j++)
    {
      h_index index_j = h_index (this, self, j, i), index_bj = h_index (index_b, y_offsets[j], 0, index_j.getNy(), index_b -> getNx());
      h_ops_tree * op_j = B -> generateOps_GEMM(&index_bj, &elements[j * nx + i], &index_j, B, &index_bi, tmp_mngr);
      op -> setChild(op_j, child_offset[i] + j - i);
      delete op_j;
    }
  }

  delete[] child_offset;
  return op;
}

h_ops_tree * Hierarchical::generateOps_TRSML (const h_index * self, const LowRank * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  h_ops_tree * op = new h_ops_tree (trsml, index_b, self);

  int n = nx > ny ? ny : nx, * child_offset = new int[(size_t) n + 1];
  child_offset[0] = 0;

  for (int i = 1; i <= n; i++)
  { child_offset[i] = child_offset[i - 1] + ny - i + 1; }

  op -> resizeChildren(child_offset[n]);

#pragma omp parallel for if (omp_in_parallel() == 0)
  for (int i = 0; i < n; i++)
  {
    h_index index_i = h_index (this, self, i, i), index_biu = h_index (index_b, y_offsets[i], 0, index_i.getNy(), index_b -> getNx());
    index_biu.setU();
    h_ops_tree * op_i = elements[i * nx + i].generateOps_TRSML(&index_i, B, &index_biu, tmp_mngr);
    op -> setChild(op_i, child_offset[i]);
    delete op_i;

    for (int j = i + 1; j < ny; j++)
    {
      h_index index_j = h_index (this, self, j, i), index_bju = h_index (index_b, y_offsets[j], 0, index_j.getNy(), index_b -> getNx());
      index_bju.setU();
      h_ops_tree * op_j = B -> generateOps_GEMM(&index_bju, &elements[j * nx + i], &index_j, B, &index_biu, tmp_mngr);
      op -> setChild(op_j, child_offset[i] + j - i);
      delete op_j;
    }
  }

  delete[] child_offset;
  return op;
}

h_ops_tree * Hierarchical::generateOps_TRSML (const h_index * self, const Hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  if (ny != B -> ny) 
  { printf("Matrices are partitioned differently in H-H TRSML.\n"); return nullptr; }

  h_ops_tree * op = new h_ops_tree (trsml, index_b, self);

  int n = nx > ny ? ny : nx, * child_offset = new int[(size_t) n + 1];
  child_offset[0] = 0;

  for (int i = 1; i <= n; i++)
  { child_offset[i] = child_offset[i - 1] + (B -> nx) * (ny - i + 1); }

  op -> resizeChildren(child_offset[n]);

#pragma omp parallel for if (omp_in_parallel() == 0)
  for (int i = 0; i < n; i++)
  {
    h_index index_i = h_index (this, self, i, i);

    for (int j = 0; j < B -> nx; j++)
    {
      h_index index_bj = h_index (B, index_b, i, j);

      h_ops_tree * op_j = elements[i * nx + i].generateOps_TRSML(&index_i, &(B -> elements)[i * (B -> nx) + j], &index_bj, tmp_mngr);
      op -> setChild(op_j, child_offset[i] + j);
      delete op_j;

      for (int k = i + 1; k < ny; k++)
      {
        h_index index_k = h_index (this, self, k, i), index_bk = h_index (B, index_b, k, j);
        h_ops_tree * op_k = (B -> elements[k * (B -> nx) + j]).generateOps_GEMM(&index_bk, &elements[k * nx + i], &index_k, &(B -> elements)[i * (B -> nx) + j], &index_bj, tmp_mngr);
        op -> setChild(op_k, child_offset[i] + (k - i) * B -> nx + j);
        delete op_k;
      }
    }
  }

  delete[] child_offset;
  return op;
}

h_ops_tree * Hierarchical::generateOps_TRSML (const h_index * self, const Element * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  const Hierarchical *h_b = B -> getElementHierarchical();
  const LowRank *lr_b = B -> getElementLowRank();
  const Dense *d_b = B -> getElementDense();

  if (d_b != nullptr)
  { return generateOps_TRSML (self, d_b, index_b, tmp_mngr); }
  if (lr_b != nullptr)
  { return generateOps_TRSML (self, lr_b, index_b, tmp_mngr); }
  if (h_b != nullptr)
  { return generateOps_TRSML (self, h_b, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * Hierarchical::generateOps_TRSMR (const h_index * self, const Dense * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  h_ops_tree * op = new h_ops_tree (trsmr, index_b, self);

  int n = nx > ny ? ny : nx, * child_offset = new int[(size_t) n + 1];
  child_offset[0] = 0;

  for (int i = 1; i <= n; i++)
  { child_offset[i] = child_offset[i - 1] + nx - i + 1; }

  op -> resizeChildren(child_offset[n]);

#pragma omp parallel for if (omp_in_parallel() == 0)
  for (int i = 0; i < n; i++)
  {
    h_index index_i = h_index (this, self, i, i), index_bi = h_index (index_b, 0, x_offsets[i], index_b -> getNy(), index_i.getNx());
    h_ops_tree * op_i = elements[i * nx + i].generateOps_TRSMR(&index_i, B, &index_bi, tmp_mngr);
    op -> setChild(op_i, child_offset[i]);
    delete op_i;

    for (int j = i + 1; j < nx; j++)
    {
      h_index index_j = h_index (this, self, i, j), index_bj = h_index (index_b, 0, x_offsets[j], index_b -> getNy(), index_j.getNx());
      h_ops_tree * op_j = B -> generateOps_GEMM(&index_bj, &elements[j * nx + i], &index_j, B, &index_bi, tmp_mngr);
      op -> setChild(op_j, child_offset[i] + j - i);
      delete op_j;
    }
  }

  delete[] child_offset;
  return op;
}

h_ops_tree * Hierarchical::generateOps_TRSMR (const h_index * self, const LowRank * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  h_ops_tree * op = new h_ops_tree (trsmr, index_b, self);

  int n = nx > ny ? ny : nx, * child_offset = new int[(size_t) n + 1];
  child_offset[0] = 0;

  for (int i = 1; i <= n; i++)
  { child_offset[i] = child_offset[i - 1] + nx - i + 1; }

  op -> resizeChildren(child_offset[n]);

#pragma omp parallel for if (omp_in_parallel() == 0)
  for (int i = 0; i < n; i++)
  {
    h_index index_i = h_index (this, self, i, i), index_biv = h_index (index_b, 0, x_offsets[i], index_b -> getNy(), index_i.getNx());
    index_biv.setVT();
    h_ops_tree * op_i = elements[i * nx + i].generateOps_TRSMR(&index_i, B, &index_biv, tmp_mngr);
    op -> setChild(op_i, child_offset[i]);
    delete op_i;

    for (int j = i + 1; j < nx; j++)
    {
      h_index index_j = h_index (this, self, i, j), index_bjv = h_index (index_b, 0, x_offsets[j], index_b -> getNy(), index_j.getNx());
      index_bjv.setVT();
      h_ops_tree * op_j = B -> generateOps_GEMM(&index_bjv, B, &index_biv, &elements[i * nx + j], &index_j, tmp_mngr);
      op -> setChild(op_j, child_offset[i] + j - i);
      delete op_j;
    }
  }

  delete[] child_offset;
  return op;
}

h_ops_tree * Hierarchical::generateOps_TRSMR (const h_index * self, const Hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  if (nx != B -> nx)
  { printf("Matrices are partitioned differently in H-H TRSMR.\n"); return nullptr; }

  h_ops_tree * op = new h_ops_tree (trsmr, index_b, self);

  int n = nx > ny ? ny : nx, * child_offset = new int[(size_t) n + 1];
  child_offset[0] = 0;

  for (int i = 1; i <= n; i++)
  { child_offset[i] = child_offset[i - 1] + (B -> ny) * (nx - i + 1); }

  op -> resizeChildren(child_offset[n]);

#pragma omp parallel for if (omp_in_parallel() == 0)
  for (int i = 0; i < n; i++)
  {
    h_index index_i = h_index (this, self, i, i);

    for (int j = 0; j < B -> ny; j++)
    {
      h_index index_bj = h_index (B, index_b, j, i);

      h_ops_tree * op_j = elements[i * nx + i].generateOps_TRSMR(&index_i, &(B -> elements)[j * (B -> nx) + i], &index_bj, tmp_mngr);
      op -> setChild(op_j, child_offset[i] + j);
      delete op_j;

      for (int k = i + 1; k < nx; k++)
      {
        h_index index_k = h_index (this, self, i, k), index_bk = h_index (B, index_b, j, k);
        h_ops_tree * op_k = (B -> elements[j * (B -> nx) + k]).generateOps_GEMM(&index_bk, &(B -> elements)[j * (B -> nx) + i], &index_bj, &elements[i * nx + k], &index_k, tmp_mngr);
        op -> setChild(op_k, child_offset[i] + (k - i) * B -> ny + j);
        delete op_k;
      }
    }
  }

  delete[] child_offset;
  return op;
}

h_ops_tree * Hierarchical::generateOps_TRSMR (const h_index * self, const Element * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  const Hierarchical *h_b = B -> getElementHierarchical();
  const LowRank *lr_b = B -> getElementLowRank();
  const Dense *d_b = B -> getElementDense();

  if (d_b != nullptr)
  { return generateOps_TRSMR (self, d_b, index_b, tmp_mngr); }
  if (lr_b != nullptr)
  { return generateOps_TRSMR (self, lr_b, index_b, tmp_mngr); }
  if (h_b != nullptr)
  { return generateOps_TRSMR (self, h_b, index_b, tmp_mngr); }

  return nullptr;  
}

h_ops_tree * Hierarchical::generateOps_ACCM (const h_index * self, const h_index * index_tmp_lr) const
{
  h_ops_tree * op = new h_ops_tree (accum, self, index_tmp_lr);
  op -> resizeChildren(nx * ny);

#pragma omp parallel for if (omp_in_parallel() == 0)
  for (int i = 0; i < ny * nx; i++)
  {
    const int row = i / nx, col = i - row * nx;
    const h_index index_m = h_index (this, self, row, col), index_lr = h_index (index_tmp_lr, y_offsets[row], x_offsets[col], index_m.getNy(), index_m.getNx());
    h_ops_tree * op_i = elements[i].generateOps_ACCM(&index_m, &index_lr);
    op -> setChild(op_i, i);
    delete op_i;
  }

  return op;  
}

h_ops_tree * Hierarchical::generateOps_GEMM (const h_index * self, const Dense * A, const h_index * index_a, const Dense * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b);
  op -> resizeChildren(nx * ny);

  const int k = index_a -> getNx(index_b -> getNy());

#pragma omp parallel for if (omp_in_parallel() == 0)
  for (int i = 0; i < ny * nx; i++)
  {
    const int row = i / nx, col = i - row * nx;
    const h_index index_m = h_index (this, self, row, col), index_ai = h_index (index_a, y_offsets[row], 0, index_m.getNy(), k), index_bj = h_index (index_b, 0, x_offsets[col], k, index_m.getNx());
    h_ops_tree * op_i = elements[i].generateOps_GEMM(&index_m, A, &index_ai, B, &index_bj, tmp_mngr);
    op -> setChild(op_i, i);
    delete op_i;
  }

  return op;
}

h_ops_tree * Hierarchical::generateOps_GEMM (const h_index * self, const LowRank * A, const h_index * index_a, const Dense * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b), * op_;
  op -> resizeChildren (2);

  int rank_a = index_a -> getRank(), tmp_size = rank_a * index_b -> getNx(), block_id;
#pragma omp critical
  { block_id = tmp_mngr -> requestTemp(tmp_size); }

  h_index index_tmp = h_index (self), index_av = h_index (index_a);
  index_tmp.setTemp_Low_Rank (block_id, rank_a);
  index_tmp.setU_data (index_a);

  op_ = generateOps_ACCM (self, &index_tmp);
  op -> setChild(op_, 1);
  delete op_;

  index_tmp.setVT();
  index_av.setVT();

  op_ = new h_ops_tree (gemm, &index_tmp, &index_av, index_b);
  op -> setChild (op_, 0);
  delete op_;

  return op;
}

h_ops_tree * Hierarchical::generateOps_GEMM (const h_index * self, const Hierarchical * A, const h_index * index_a, const Dense * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  if (ny != A -> ny)
  { printf("Matrices are partitioned differently in H-H.D GEMM.\n"); return nullptr; }

  h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b);
  op -> resizeChildren(nx * ny * A -> nx);

#pragma omp parallel for if (omp_in_parallel() == 0)
  for (int i = 0; i < ny * nx; i++)
  {
    const int row = i / nx, col = i - row * nx;
    const h_index index_m = h_index (this, self, row, col);
    for (int k = 0; k < A -> nx; k++)
    {
      const h_index index_ak = h_index (A, index_a, row, k), index_bk = h_index (index_b, (A -> x_offsets)[k], x_offsets[col], index_ak.getNx(), index_m.getNx());
      h_ops_tree * op_k = elements[i].generateOps_GEMM(&index_m, &(A -> elements[row * (A -> nx) + k]), &index_ak, B, &index_bk, tmp_mngr);
      op -> setChild(op_k, i * (A -> nx) + k);
      delete op_k;
    }
  }

  return op;
}

h_ops_tree * Hierarchical::generateOps_GEMM (const h_index * self, const Element * A, const h_index * index_a, const Dense * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  const Hierarchical *h_a = A -> getElementHierarchical();
  const LowRank *lr_a = A -> getElementLowRank();
  const Dense *d_a = A -> getElementDense();

  if (d_a != nullptr)
  { return generateOps_GEMM (self, d_a, index_a, B, index_b, tmp_mngr); }
  if (lr_a != nullptr)
  { return generateOps_GEMM (self, lr_a, index_a, B, index_b, tmp_mngr); }
  if (h_a != nullptr)
  { return generateOps_GEMM (self, h_a, index_a, B, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * Hierarchical::generateOps_GEMM (const h_index * self, const Dense * A, const h_index * index_a, const LowRank * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b), * op_;

  op -> resizeChildren (2);

  int rank_b = index_b -> getRank(), tmp_size = rank_b * index_a -> getNy(), block_id;
#pragma omp critical
  { block_id = tmp_mngr -> requestTemp(tmp_size); }

  h_index index_tmp = h_index (self), index_bu = h_index (index_b);
  index_tmp.setTemp_Low_Rank (block_id, rank_b);
  index_tmp.setVT_data (index_b);

  op_ = generateOps_ACCM (self, &index_tmp);
  op -> setChild (op_, 1);
  delete op_;

  index_tmp.setU();
  index_bu.setU();

  op_ = new h_ops_tree (gemm, &index_tmp, index_a, &index_bu);
  op -> setChild (op_, 0);
  delete op_;

  return op;
}

h_ops_tree * Hierarchical::generateOps_GEMM (const h_index * self, const LowRank * A, const h_index * index_a, const LowRank * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b), * op_;

  op -> resizeChildren (2);

  h_index index_tmp = h_index (self);
  bool a; 
  int rank = index_a -> getMinRank (index_b, &a), tmp_size = rank * (a ? index_b -> getNx() : index_a -> getNy()), block_id;

#pragma omp critical
  { block_id = tmp_mngr -> requestTemp(tmp_size); }

  index_tmp.setTemp_Low_Rank(block_id, rank);
  if (a)
  { index_tmp.setU_data(index_a); }
  else
  { index_tmp.setVT_data(index_b); }

  op_ = generateOps_ACCM (self, &index_tmp);
  op -> setChild (op_, 1);
  delete op_;

  if (a)
  {
    h_index index_av = h_index (index_a);
    index_tmp.setVT();
    index_av.setVT();

    op_ = new h_ops_tree (gemm, &index_tmp, &index_av, index_b);
  }
  else
  {
    h_index index_bu = h_index (index_b);
    index_tmp.setU();
    index_bu.setU();

    op_ = new h_ops_tree (gemm, &index_tmp, index_a, &index_bu);
  }

  op -> setChild (op_, 0);
  delete op_;

  return op;
}

h_ops_tree * Hierarchical::generateOps_GEMM (const h_index * self, const Hierarchical * A, const h_index * index_a, const LowRank * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  if (ny != A -> ny)
  { printf("Matrices are partitioned differently in H-H.LR GEMM.\n"); return nullptr; }

  h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b);

  op -> resizeChildren (2);

  int rank_b = index_b -> getRank(), tmp_size = rank_b * index_a -> getNy(self -> getNy()), block_id;

#pragma omp critical
  { block_id = tmp_mngr -> requestTemp(tmp_size); }

  h_index index_tmp = h_index (self), index_bu = h_index (index_b);
  index_tmp.setTemp_Low_Rank (block_id, rank_b);
  index_tmp.setVT_data (index_b);

  h_ops_tree * op_ = generateOps_ACCM (self, &index_tmp);
  op -> setChild (op_, 1);
  delete op_;

  index_tmp.setU();
  index_bu.setU();

  op_ = LowRank :: generateOps_GEMM (&index_tmp, A, index_a, B, &index_bu, tmp_mngr);
  op -> setChild (op_, 0);
  delete op_;

  return op;
}

h_ops_tree * Hierarchical::generateOps_GEMM (const h_index * self, const Element * A, const h_index *index_a, const LowRank * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  const Hierarchical *h_a = A -> getElementHierarchical();
  const LowRank *lr_a = A -> getElementLowRank();
  const Dense *d_a = A -> getElementDense();

  if (d_a != nullptr)
  { return generateOps_GEMM (self, d_a, index_a, B, index_b, tmp_mngr); }
  if (lr_a != nullptr)
  { return generateOps_GEMM (self, lr_a, index_a, B, index_b, tmp_mngr); }
  if (h_a != nullptr)
  { return generateOps_GEMM (self, h_a, index_a, B, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * Hierarchical::generateOps_GEMM (const h_index * self, const Dense * A, const h_index * index_a, const Hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  if (nx != B -> nx)
  { printf("Matrices are partitioned differently in H-D.H GEMM.\n"); return nullptr; }

  h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b);
  op -> resizeChildren (nx * ny * B -> ny);

#pragma omp parallel for if (omp_in_parallel() == 0)
  for (int i = 0; i < ny * nx; i++)
  {
    const int row = i / nx, col = i - row * nx;
    const h_index index_m = h_index (this, self, row, col);
    for (int k = 0; k < B -> ny; k++)
    {
      const h_index index_bk = h_index (B, index_b, k, col), index_ak = h_index (index_a, y_offsets[row], (B -> y_offsets)[k], index_m.getNy(), index_bk.getNy());
      h_ops_tree * op_k = elements[i].generateOps_GEMM(&index_m, A, &index_ak, &(B -> elements[k * (B -> nx) + col]), &index_bk, tmp_mngr);
      op -> setChild(op_k, i * (B -> ny) + k);
      delete op_k;
    }
  }

  return op;  
}

h_ops_tree * Hierarchical::generateOps_GEMM (const h_index * self, const LowRank * A, const h_index * index_a, const Hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  if (nx != B -> nx)
  { printf("Matrices are partitioned differently in H-LR.H GEMM.\n"); return nullptr; }

  h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b);
  op -> resizeChildren (2);

  int rank_a = index_a -> getRank(), tmp_size = rank_a * index_b -> getNx(self -> getNx()), block_id;
#pragma omp critical
  { block_id = tmp_mngr -> requestTemp(tmp_size); }

  h_index index_tmp = h_index (self), index_av = h_index (index_a);
  index_tmp.setTemp_Low_Rank (block_id, rank_a);
  index_tmp.setU_data (index_a);

  h_ops_tree * op_ = generateOps_ACCM (self, &index_tmp);
  op -> setChild(op_, 1);
  delete op_;

  index_tmp.setVT();
  index_av.setVT();

  op_ = LowRank :: generateOps_GEMM (&index_tmp, A, &index_av, B, index_b, tmp_mngr);
  op -> setChild (op_, 0);
  delete op_;

  return op;
}

h_ops_tree * Hierarchical::generateOps_GEMM (const h_index * self, const Hierarchical * A, const h_index * index_a, const Hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  if (ny != A -> ny || nx != B -> nx || A -> nx != B -> ny)
  { printf("Partition error in H-H.H GEMM.\n"); return nullptr; }

  h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b);
  op -> resizeChildren(ny * nx * A -> nx);

#pragma omp parallel for if (omp_in_parallel() == 0)
  for (int i = 0; i < ny * nx; i++)
  {
    const int row = i / nx, col = i - row * nx;
    const h_index index_m = h_index (this, self, row, col);

    for (int k = 0; k < A -> nx; k++)
    {
      const h_index index_ak = h_index (A, index_a, row, k), index_bk = h_index (B, index_b, k, col);

      h_ops_tree * op_k = elements[i].generateOps_GEMM (&index_m, &(A -> elements)[row * (A -> nx) + k], &index_ak, &(B -> elements)[k * (B -> nx) + col], &index_bk, tmp_mngr);
      op -> setChild(op_k, i * A -> nx + k);
      delete op_k;
    }
  }

  return op;
}

h_ops_tree * Hierarchical::generateOps_GEMM (const h_index * self, const Element * A, const h_index * index_a, const Hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  const Hierarchical *h_a = A -> getElementHierarchical();
  const LowRank *lr_a = A -> getElementLowRank();
  const Dense *d_a = A -> getElementDense();

  if (d_a != nullptr)
  { return generateOps_GEMM (self, d_a, index_a, B, index_b, tmp_mngr); }
  if (lr_a != nullptr)
  { return generateOps_GEMM (self, lr_a, index_a, B, index_b, tmp_mngr); }
  if (h_a != nullptr)
  { return generateOps_GEMM (self, h_a, index_a, B, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * Hierarchical::generateOps_GEMM (const h_index * self, const Dense * A, const h_index * index_a, const Element * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  const Hierarchical *h_b = B -> getElementHierarchical();
  const LowRank *lr_b = B -> getElementLowRank();
  const Dense *d_b = B -> getElementDense();

  if (d_b != nullptr)
  { return generateOps_GEMM (self, A, index_a, d_b, index_b, tmp_mngr); }
  if (lr_b != nullptr)
  { return generateOps_GEMM (self, A, index_a, lr_b, index_b, tmp_mngr); }
  if (h_b != nullptr)
  { return generateOps_GEMM (self, A, index_a, h_b, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * Hierarchical::generateOps_GEMM (const h_index * self, const LowRank * A, const h_index * index_a, const Element * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  const Hierarchical *h_b = B -> getElementHierarchical();
  const LowRank *lr_b = B -> getElementLowRank();
  const Dense *d_b = B -> getElementDense();

  if (d_b != nullptr)
  { return generateOps_GEMM (self, A, index_a, d_b, index_b, tmp_mngr); }
  if (lr_b != nullptr)
  { return generateOps_GEMM (self, A, index_a, lr_b, index_b, tmp_mngr); }
  if (h_b != nullptr)
  { return generateOps_GEMM (self, A, index_a, h_b, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * Hierarchical::generateOps_GEMM (const h_index * self, const Hierarchical * A, const h_index * index_a, const Element * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  const Hierarchical *h_b = B -> getElementHierarchical();
  const LowRank *lr_b = B -> getElementLowRank();
  const Dense *d_b = B -> getElementDense();

  if (d_b != nullptr)
  { return generateOps_GEMM (self, A, index_a, d_b, index_b, tmp_mngr); }
  if (lr_b != nullptr)
  { return generateOps_GEMM (self, A, index_a, lr_b, index_b, tmp_mngr); }
  if (h_b != nullptr)
  { return generateOps_GEMM (self, A, index_a, h_b, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * Hierarchical::generateOps_GEMM (const h_index * self, const Element * A, const h_index * index_a, const Element * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  const Hierarchical *h_b = B -> getElementHierarchical();
  const LowRank *lr_b = B -> getElementLowRank();
  const Dense *d_b = B -> getElementDense();

  if (d_b != nullptr)
  { return generateOps_GEMM (self, A, index_a, d_b, index_b, tmp_mngr); }
  if (lr_b != nullptr)
  { return generateOps_GEMM (self, A, index_a, lr_b, index_b, tmp_mngr); }
  if (h_b != nullptr)
  { return generateOps_GEMM (self, A, index_a, h_b, index_b, tmp_mngr); }

  return nullptr;
}*/


