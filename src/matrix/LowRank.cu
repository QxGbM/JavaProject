
#include <matrix/LowRank.cuh>

LowRank::LowRank(const int m, const int n, const int rank, const int ld) : Element (element_t::low_rank, 0, 0) {
  U = new Clusterbasis(m, rank, &(Element::abs_y), ld);
  V = new Clusterbasis(n, rank, &(Element::abs_x), ld);
  S = new Dense(rank, rank);
}

LowRank::LowRank(const int m, const int n, const int rank, const int abs_y, const int abs_x, const int ld) : Element(element_t::low_rank, abs_y, abs_x) {
  U = new Clusterbasis(m, rank, &(Element::abs_y), ld);
  V = new Clusterbasis(n, rank, &(Element::abs_x), ld);
  S = new Dense(rank, rank);
}

LowRank::~LowRank() {
  delete U;
  delete V;
  if (S != nullptr)
  { delete S; }
  Element::~Element();
}

LowRank* LowRank::getElementLowRank()
{ return this; }

int LowRank::getRowDimension() const 
{ return U->getDimension(); }

int LowRank::getColumnDimension() const
{ return V->getDimension(); }

int LowRank::getRank () const {
  using std::min;
  return min(U->getRank(), V->getRank()); 
}

Dense* LowRank::convertToDense() const {
  return nullptr;
}

void LowRank::print() const {
  
}

void LowRank::print(vector<int>& indices, vector<int>& config) const {

}

Clusterbasis* LowRank::getU() {
  return U;
}

Clusterbasis* LowRank::getV() {
  return V;
}


/*
h_ops_tree * LowRank::generateOps_GETRF (const h_index * self, dev_temp * tmp_mngr)
{ 
  printf("Error: GETRF should not be performed on low-rank matrices.\n");
  return nullptr;
}

h_ops_tree * LowRank::generateOps_TRSML (const h_index * self, const Dense * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  printf("Error: TRSML should not have a low-rank matrix be the lower triangular.\n");
  return nullptr;
}

h_ops_tree * LowRank::generateOps_TRSML (const h_index * self, const LowRank * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  printf("Error: TRSML should not have a low-rank matrix be the lower triangular.\n");
  return nullptr;
}

h_ops_tree * LowRank::generateOps_TRSML (const h_index * self, const Hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  printf("Error: TRSM should not have a low-rank matrix be the lower triangular.\n");
  return nullptr;
}

h_ops_tree * LowRank::generateOps_TRSML (const h_index * self, const Element * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  printf("Error: TRSM should not have a low-rank matrix be the lower triangular.\n");
  return nullptr;
}

h_ops_tree * LowRank::generateOps_TRSMR (const h_index * self, const Dense * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  printf("Error: TRSMR should not have a low-rank matrix be the upper triangular.\n");
  return nullptr;
}

h_ops_tree * LowRank::generateOps_TRSMR (const h_index * self, const LowRank * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  printf("Error: TRSMR should not have a low-rank matrix be the upper triangular.\n");
  return nullptr;
}

h_ops_tree * LowRank::generateOps_TRSMR (const h_index * self, const Hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  printf("Error: TRSMR should not have a low-rank matrix be the upper triangular.\n");
  return nullptr;
}

h_ops_tree * LowRank::generateOps_TRSMR (const h_index * self, const Element * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  printf("Error: TRSMR should not have a low-rank matrix be the upper triangular.\n");
  return nullptr;
}

h_ops_tree * LowRank::generateOps_ACCM (const h_index * self, const h_index * index_tmp_lr)
{
  return new h_ops_tree (accum, self, index_tmp_lr); 
}

h_ops_tree * LowRank::generateOps_GEMM (const h_index * self, const Dense * A, const h_index * index_a, const Dense * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b);

  if (!self -> isLowRank_Full())
  {
    printf("WARNING: Potential Accuracy Loss from an attempt to accumulate Dense into Low-Rank.\n");
    op -> resizeChildren (2);

    int block_id, tmp_size = self -> getSize();
#pragma omp critical
    { block_id = tmp_mngr -> requestTemp(tmp_size); }

    h_index index_tmp = h_index (self); index_tmp.setTemp_Dense (block_id);

    h_ops_tree * op_ = new h_ops_tree (gemm, &index_tmp, index_a, index_b);
    op -> setChild (op_, 0);
    delete op_;

    op_ = new h_ops_tree (accum, self, &index_tmp);
    op -> setChild (op_, 1);
    delete op_;
  }

  return op;
}

h_ops_tree * LowRank::generateOps_GEMM (const h_index * self, const LowRank * A, const h_index * index_a, const Dense * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b), * op_;

  if (!(self -> isVT() && index_a -> isVT()))
  {
    op -> resizeChildren (2);

    int rank_a = index_a -> getRank(), tmp_size = rank_a * index_b -> getNx(self -> getNx()), block_id;
#pragma omp critical
    { block_id = tmp_mngr -> requestTemp(tmp_size); }

    h_index index_tmp = h_index (self), index_av = h_index (index_a);
    index_tmp.setTemp_Low_Rank (block_id, rank_a);
    index_tmp.setU_data (index_a);

    op_ = new h_ops_tree (accum, self, &index_tmp);
    op -> setChild(op_, 1);
    delete op_;

    index_tmp.setVT();
    index_av.setVT();

    op_ = new h_ops_tree (gemm, &index_tmp, &index_av, index_b);
    op -> setChild (op_, 0);
    delete op_;
  }

  return op;
}

h_ops_tree * LowRank::generateOps_GEMM (const h_index * self, const Hierarchical * A, const h_index * index_a, const Dense * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b);

  const int n_k = A -> getNx_blocks(), n_mk = n_k * A -> getNy_blocks();
  int * y, * k, x = self -> getNx(index_b -> getNx());
  A -> getOffsets_y(&y);
  A -> getOffsets_x(&k);

  if (self -> isLowRank_Full())
  {
    printf("WARNING: Potential Accuracy Loss from an attempt to accumulate Dense into Low-Rank.\n");

    op -> resizeChildren(n_mk + 1);

    int block_id, tmp_size = self -> getSize();
#pragma omp critical
    { block_id = tmp_mngr -> requestTemp(tmp_size); }

    h_index index_tmp = h_index (self); index_tmp.setTemp_Dense(block_id);

    h_ops_tree * op_ = new h_ops_tree (accum, self, &index_tmp);
    op -> setChild (op_, n_mk);
    delete op_;

#pragma omp parallel for if (omp_in_parallel() == 0)
    for (int i = 0; i < n_mk; i++)
    {
      const int row = i / n_k, col = i - row * n_k;
      const h_index index_ai = h_index (A, index_a, row, col), index_m = h_index (&index_tmp, y[row], 0, index_ai.getNy(), x), index_bj = h_index (index_b, k[col], 0, index_ai.getNx(), x);
      h_ops_tree * op_i = Dense::generateOps_GEMM(&index_m, A -> getElement_blocks(row, col), &index_ai, B, &index_bj, tmp_mngr);
      op -> setChild(op_i, i);
      delete op_i;
    }
  }
  else
  {
    op -> resizeChildren(n_mk);

#pragma omp parallel for if (omp_in_parallel() == 0)
    for (int i = 0; i < n_mk; i++)
    {
      const int row = i / n_k, col = i - row * n_k;
      const h_index index_ai = h_index (A, index_a, row, col), index_m = h_index (self, y[row], 0, index_ai.getNy(), x), index_bj = h_index (index_b, k[col], 0, index_ai.getNx(), x);
      h_ops_tree * op_i = generateOps_GEMM(&index_m, A -> getElement_blocks(row, col), &index_ai, B, &index_bj, tmp_mngr);
      op -> setChild(op_i, i);
      delete op_i;
    }
  }

  delete[] y;
  delete[] k;
  return op;
}

h_ops_tree * LowRank::generateOps_GEMM (const h_index * self, const Element * A, const h_index * index_a, const Dense * B, const h_index * index_b, dev_temp * tmp_mngr)
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

h_ops_tree * LowRank::generateOps_GEMM (const h_index * self, const Dense * A, const h_index * index_a, const LowRank * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b);

  if (!(self -> isU() && index_b -> isU()))
  {
    op -> resizeChildren (2);
    int rank_b = index_b -> getRank(), tmp_size = rank_b * index_a -> getNy(self -> getNy()), block_id;
#pragma omp critical
    { block_id = tmp_mngr -> requestTemp(tmp_size); }

    h_index index_tmp = h_index (self), index_bu = h_index (index_b);
    index_tmp.setTemp_Low_Rank (block_id, rank_b);
    index_tmp.setVT_data (index_b);

    h_ops_tree * op_ = new h_ops_tree (accum, self, &index_tmp);
    op -> setChild (op_, 1);
    delete op_;

    index_tmp.setU();
    index_bu.setU();

    op_ = new h_ops_tree (gemm, &index_tmp, index_a, &index_bu);
    op -> setChild (op_, 0);
    delete op_;
  }

  return op;
}

h_ops_tree * LowRank::generateOps_GEMM (const h_index * self, const LowRank * A, const h_index * index_a, const LowRank * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b);

  if (self -> isLowRank_Full())
  {
    op -> resizeChildren (2);

    h_index index_tmp = h_index (self);
    bool a; 
    int rank = index_a -> getMinRank (index_b, &a);
    int tmp_size = rank * (a ? index_b -> getNx(self -> getNx()) : index_a -> getNy(self -> getNy()));
    int block_id;

#pragma omp critical
    { block_id = tmp_mngr -> requestTemp(tmp_size); }

    index_tmp.setTemp_Low_Rank(block_id, rank);
    if (a)
    { index_tmp.setU_data(index_a); }
    else
    { index_tmp.setVT_data(index_b); }

    h_ops_tree * op_ = new h_ops_tree (accum, self, &index_tmp);
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
  }

  return op;
}

h_ops_tree * LowRank::generateOps_GEMM (const h_index * self, const Hierarchical * A, const h_index * index_a, const LowRank * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b);

  const int n_k = A -> getNx_blocks(), n_mk = n_k * A -> getNy_blocks();
  int * y, * k, x = index_b -> getNx(self -> getNx());
  A -> getOffsets_y(&y);
  A -> getOffsets_x(&k);

  if (self -> isU() && index_b -> isU())
  {
      op -> resizeChildren (n_mk);

#pragma omp parallel for if (omp_in_parallel() == 0)
    for (int i = 0; i < n_mk; i++)
    {
      const int row = i / n_k, col = i - row * n_k;
      const h_index index_ai = h_index (A, index_a, row, col), index_m = h_index (self, y[row], 0, index_ai.getNy(), x), index_bj = h_index (index_b, k[col], 0, index_ai.getNx(), x);
      h_ops_tree * op_i = generateOps_GEMM(&index_m, A -> getElement_blocks(row, col), &index_ai, B, &index_bj, tmp_mngr);
      op -> setChild(op_i, i);
      delete op_i;
    }
  }
  else
  {
    op -> resizeChildren (n_mk + 1);

    int rank_b = index_b -> getRank(), tmp_size = rank_b * index_a -> getNy(self -> getNy()), block_id;
#pragma omp critical
    { block_id = tmp_mngr -> requestTemp(tmp_size); }

    h_index index_tmp = h_index (self), index_bu = h_index (index_b); 
    index_tmp.setTemp_Low_Rank (block_id, rank_b);
    index_tmp.setVT_data (index_b);

    h_ops_tree * op_ = new h_ops_tree (accum, self, &index_tmp);
    op -> setChild (op_, n_mk);
    delete op_;

    index_tmp.setU();
    index_bu.setU();

#pragma omp parallel for if (omp_in_parallel() == 0)
    for (int i = 0; i < n_mk; i++)
    {
      const int row = i / n_k, col = i - row * n_k;
      const h_index index_ai = h_index (A, index_a, row, col), index_m = h_index (&index_tmp, y[row], 0, index_ai.getNy(), x), index_bj = h_index (&index_bu, k[col], 0, index_ai.getNx(), x);
      h_ops_tree * op_i = generateOps_GEMM(&index_m, A -> getElement_blocks(row, col), &index_ai, B, &index_bj, tmp_mngr);
      op -> setChild(op_i, i);
      delete op_i;
    }
  }

  delete[] y;
  delete[] k;
  return op;
}

h_ops_tree * LowRank::generateOps_GEMM (const h_index * self, const Element * A, const h_index * index_a, const LowRank * B, const h_index * index_b, dev_temp * tmp_mngr)
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

h_ops_tree * LowRank::generateOps_GEMM (const h_index * self, const Dense * A, const h_index * index_a, const Hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b);

  const int n_n = B -> getNx_blocks(), n_nk = n_n * B -> getNy_blocks();
  int * x, * k, y = self -> getNy(index_a -> getNy());
  B -> getOffsets_y(&k);
  B -> getOffsets_x(&x);

  if (self -> isLowRank_Full())
  {
    printf("WARNING: Potential Accuracy Loss from an attempt to accumulate Dense into Low-Rank.\n");

    op -> resizeChildren (n_nk + 1);

    int block_id, tmp_size = self -> getSize();
#pragma omp critical
    { block_id = tmp_mngr -> requestTemp(tmp_size); }

    h_index index_tmp = h_index (self); index_tmp.setTemp_Dense (block_id);

    h_ops_tree * op_ = new h_ops_tree (accum, self, &index_tmp);
    op -> setChild (op_, n_nk);
    delete op_;

#pragma omp parallel for if (omp_in_parallel() == 0)
    for (int i = 0; i < n_nk; i++)
    {
      const int row = i / n_n, col = i - row * n_n;
      const h_index index_bj = h_index (B, index_b, row, col), index_m = h_index (&index_tmp, 0, x[col], y, index_bj.getNx()), index_ai = h_index (index_a, 0, k[row], y, index_bj.getNy());
      h_ops_tree * op_i = Dense::generateOps_GEMM(&index_m, A, &index_ai, B -> getElement_blocks(row, col), &index_bj, tmp_mngr);
      op -> setChild(op_i, i);
      delete op_i;
    }
  }
  else
  {
    op -> resizeChildren (n_nk);

#pragma omp parallel for if (omp_in_parallel() == 0)
    for (int i = 0; i < n_nk; i++)
    {
      const int row = i / n_n, col = i - row * n_n;
      const h_index index_bj = h_index (B, index_b, row, col), index_m = h_index (self, 0, x[col], y, index_bj.getNx()), index_ai = h_index (index_a, 0, k[row], y, index_bj.getNy());
      h_ops_tree * op_i = generateOps_GEMM(&index_m, A, &index_ai, B -> getElement_blocks(row, col), &index_bj, tmp_mngr);
      op -> setChild(op_i, i);
      delete op_i;
    }
  }

  delete[] x;
  delete[] k;
  return op;
}

h_ops_tree * LowRank::generateOps_GEMM (const h_index * self, const LowRank * A, const h_index * index_a, const Hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b);

  const int n_n = B -> getNx_blocks(), n_nk = n_n * B -> getNy_blocks();
    
  int * x, * k, y = self -> getNy(index_a -> getNy());
  B -> getOffsets_x(&x);
  B -> getOffsets_y(&k);

  if (self -> isVT() && index_a -> isVT())
  {
    op -> resizeChildren(n_nk);

#pragma omp parallel for if (omp_in_parallel() == 0)
    for (int i = 0; i < n_nk; i++)
    {
      const int row = i / n_n, col = i - row * n_n;
      const h_index index_bj = h_index (B, index_b, row, col), index_m = h_index (self, 0, x[col], y, index_bj.getNx()), index_ai = h_index (index_a, 0, k[row], y, index_bj.getNy());
      h_ops_tree * op_i = generateOps_GEMM(&index_m, A, &index_ai, B -> getElement_blocks(row, col), &index_bj, tmp_mngr);
      op -> setChild(op_i, i);
      delete op_i;
    }
  }
  else
  {
    op -> resizeChildren(n_nk + 1);

    int rank_a = index_a -> getRank(), tmp_size = rank_a * index_b -> getNx(self -> getNx()), block_id;
#pragma omp critical
    { block_id = tmp_mngr -> requestTemp(tmp_size); }

    h_index index_tmp = h_index (self), index_av = h_index (index_a); 
    index_tmp.setTemp_Low_Rank(block_id, rank_a);
    index_tmp.setU_data(index_a);

    h_ops_tree * op_ = new h_ops_tree (accum, self, &index_tmp);
    op -> setChild (op_, n_nk);
    delete op_;

    index_tmp.setVT();
    index_av.setVT();

#pragma omp parallel for if (omp_in_parallel() == 0)
    for (int i = 0; i < n_nk; i++)
    {
      const int row = i / n_n, col = i - row * n_n;
      const h_index index_bj = h_index (B, index_b, row, col), index_m = h_index (&index_tmp, 0, x[col], y, index_bj.getNx()), index_ai = h_index (&index_av, 0, k[row], y, index_bj.getNy());
      h_ops_tree * op_i = generateOps_GEMM(&index_m, A, &index_ai, B -> getElement_blocks(row, col), &index_bj, tmp_mngr);
      op -> setChild(op_i, i);
      delete op_i;
    }
  }

  delete[] x;
  delete[] k;
  return op;
}

h_ops_tree * LowRank::generateOps_GEMM (const h_index * self, const Hierarchical * A, const h_index * index_a, const Hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  const int n_k = A -> getNx_blocks(); if (n_k != B -> getNy_blocks())
  { printf("Matrices are partitioned differently in LR.H-H GEMM.\n"); return nullptr; }

  h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b);

  const int n_n = B -> getNx_blocks(), n_mn = n_n * A -> getNy_blocks(), n_mnk = n_mn * n_k;
  int * x, * y;
  A -> getOffsets_y(&y);
  B -> getOffsets_x(&x);

  if (self -> isLowRank_Full())
  {
    printf("WARNING: Potential Accuracy Loss from an attempt to accumulate Dense into Low-Rank.\n");

    op -> resizeChildren(n_mnk + 1);

    int block_id, tmp_size = self -> getSize();
#pragma omp critical
    { block_id = tmp_mngr -> requestTemp(tmp_size); }

    h_index index_tmp = h_index (self); index_tmp.setTemp_Dense (block_id);

    h_ops_tree * op_ = new h_ops_tree (accum, self, &index_tmp);
    op -> setChild (op_, n_mnk);
    delete op_;

#pragma omp parallel for if (omp_in_parallel() == 0)
    for (int i = 0; i < n_mnk; i++)
    {
      const int k = i / n_mn, crd = i - k * n_mn, row = crd / n_n, col = crd - row * n_n;

      const h_index index_ai = h_index (A, index_a, row, k), index_bj = h_index (B, index_b, k, col);
      const h_index index_m = h_index (&index_tmp, y[row], x[col], index_ai.getNy(), index_bj.getNx());
      h_ops_tree * op_k = Dense::generateOps_GEMM (&index_m, A -> getElement_blocks(row, k), &index_ai, B -> getElement_blocks(k, col), &index_bj, tmp_mngr);
      op -> setChild(op_k, i);
      delete op_k;
    }
  }
  else
  {
    op -> resizeChildren(n_mnk);

#pragma omp parallel for if (omp_in_parallel() == 0)
    for (int i = 0; i < n_mnk; i++)
    {
      const int k = i / n_mn, crd = i - k * n_mn, row = crd / n_n, col = crd - row * n_n;

      const h_index index_ai = h_index (A, index_a, row, k), index_bj = h_index (B, index_b, k, col);
      const h_index index_m = h_index (self, y[row], x[col], index_ai.getNy(), index_bj.getNx());
      h_ops_tree * op_k = generateOps_GEMM (&index_m, A -> getElement_blocks(row, k), &index_ai, B -> getElement_blocks(k, col), &index_bj, tmp_mngr);
      op -> setChild(op_k, i);
      delete op_k;
    }
  }

  delete[] x;
  delete[] y;
  return op;
}

h_ops_tree * LowRank::generateOps_GEMM (const h_index * self, const Element * A, const h_index * index_a, const Hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr)
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

h_ops_tree * LowRank::generateOps_GEMM (const h_index * self, const Dense * A, const h_index * index_a, const Element * B, const h_index * index_b, dev_temp * tmp_mngr)
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

h_ops_tree * LowRank::generateOps_GEMM (const h_index * self, const LowRank * A, const h_index * index_a, const Element * B, const h_index * index_b, dev_temp * tmp_mngr)
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

h_ops_tree * LowRank::generateOps_GEMM (const h_index * self, const Hierarchical * A, const h_index * index_a, const Element * B, const h_index * index_b, dev_temp * tmp_mngr)
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

h_ops_tree * LowRank::generateOps_GEMM (const h_index * self, const Element * A, const h_index * index_a, const Element * B, const h_index * index_b, dev_temp * tmp_mngr)
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

