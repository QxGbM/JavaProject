#ifndef _DEV_LOW_RANK_CUH
#define _DEV_LOW_RANK_CUH

#include <pspl.cuh>

template <class T> class dev_low_rank 
{
private:
  int nx;
  int ny;
  int rank;
  dev_dense <T> * UxS;
  dev_dense <T> * VT;

public:

  __host__ dev_low_rank (const int x, const int y)
  {
    nx = x;
    ny = y;
    rank = (x > y) ? y : x;

    UxS = new dev_dense <T> (rank, ny);
    VT = new dev_dense <T> (rank, nx);

    VT -> loadIdentityMatrix();

  }

  __host__ ~dev_low_rank ()
  {
    delete UxS;
    delete VT;
  }

  __host__ inline int getNx () const { return nx; }

  __host__ inline int getNy () const { return ny; }

  __host__ inline int getRank () const { return rank; }

  __host__ inline dev_dense <T> * getUxS () const { return UxS; }

  __host__ inline dev_dense <T> * getVT () const { return VT; }

  __host__ inline int getNx_UxS() const { return UxS -> getNx(); }

  __host__ inline int getNy_UxS() const { return UxS -> getNy(); }

  __host__ inline int getLd_UxS() const { return UxS -> getLd(); }

  __host__ inline int getNx_VT() const { return VT -> getNx(); }

  __host__ inline int getNy_VT() const { return VT -> getNy(); }

  __host__ inline int getLd_VT() const { return VT -> getLd(); }

  __host__ inline int getOffset_UxS (const int dense_offset = 0) const
  { return rank * (dense_offset / nx); }

  __host__ inline int getOffset_VT (const int dense_offset = 0) const 
  { return rank * (dense_offset % nx) + getNy_UxS() * getLd_UxS(); }

  __host__ inline T * getElements (const int offset = 0) const 
  { 
    return offset >= getOffset_VT() ? VT -> getElements (offset - getOffset_VT()) : UxS -> getElements(offset); 
  }

  __host__ T getElement (const int x, const int y) const
  {
    T element = 0, * row = UxS -> getElements (y * rank), * col = VT -> getElements (x * rank);
    for (int i = 0; i < rank; i++)
    { element += row[i] * col[i]; }
    return element;
  }

  __host__ void adjustRank (const int rank_in)
  {
    if (rank_in > 0 && rank_in != rank)
    {
      UxS -> resize (rank_in, ny);
      VT -> resize (rank_in, nx);
      rank = rank_in;
    }
  }

  __host__ bool sameRowBasis (const dev_low_rank <T> * lr) const
  { return lr -> UxS == UxS; }

  __host__ bool sameColBasis (const dev_low_rank <T> * lr) const
  { return lr -> VT == VT; }

  __host__ h_ops_tree * generateOps_GETRF (const h_index * self) const
  { 
    printf("Error: GETRF should not be performed on low-rank matrices.\n");
    return nullptr;
  }

  __host__ h_ops_tree * generateOps_TRSML (const h_index *self, const dev_dense <T> *B, const h_index *index_b) const
  {
    printf("Error: TRSM should not have a low-rank matrix be the lower/upper triangular.\n");
    return nullptr;
  }

  __host__ h_ops_tree * generateOps_TRSML (const h_index *self, const dev_low_rank <T> *B, const h_index *index_b) const
  {
    printf("Error: TRSM should not have a low-rank matrix be the lower/upper triangular.\n");
    return nullptr;
  }

  __host__ h_ops_tree * generateOps_TRSML (const h_index *self, const dev_hierarchical <T> *B, const h_index *index_b) const
  {
    printf("Error: TRSM should not have a low-rank matrix be the lower/upper triangular.\n");
    return nullptr;
  }

  __host__ h_ops_tree * generateOps_TRSML (const h_index *self, const dev_h_element <T> *B, const h_index *index_b) const
  {
    printf("Error: TRSM should not have a low-rank matrix be the lower/upper triangular.\n");
    return nullptr;
  }

  __host__ h_ops_tree * generateOps_TRSMR (const h_index *self, const dev_dense <T> *B, const h_index *index_b) const
  {
    printf("Error: TRSM should not have a low-rank matrix be the lower/upper triangular.\n");
    return nullptr;
  }

  __host__ h_ops_tree * generateOps_TRSMR (const h_index *self, const dev_low_rank <T> *B, const h_index *index_b) const
  {
    printf("Error: TRSM should not have a low-rank matrix be the lower/upper triangular.\n");
    return nullptr;
  }

  __host__ h_ops_tree * generateOps_TRSMR (const h_index *self, const dev_hierarchical <T> *B, const h_index *index_b) const
  {
    printf("Error: TRSM should not have a low-rank matrix be the lower/upper triangular.\n");
    return nullptr;
  }

  __host__ h_ops_tree * generateOps_TRSMR (const h_index *self, const dev_h_element <T> *B, const h_index *index_b) const
  {
    printf("Error: TRSM should not have a low-rank matrix be the lower/upper triangular.\n");
    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_dense <T> *A, const h_index *index_a, const bool A_T, const dev_dense <T> *B, const h_index *index_b, const bool B_T) const
  {
    const int x_m = getNx(), y_m = getNy(), r_m = getRank(), x_a = A -> getNx(), y_a = A -> getNy(), x_b = B -> getNx(), y_b = B -> getNy();
    const int m = y_m > y_a ? y_a : y_m, n = r_m, k = x_a > y_b ? y_b : x_a, l = x_m > x_b ? x_b : x_m;
    const int ld_m = getLd_UxS(), ld_a = A -> getLd(), ld_b = B -> getLd(), ld_c = getLd_VT();
    const h_index * index_u = self -> child_UxS(this), * index_v = self -> child_VT(this);
    h_ops_tree * op = new h_ops_tree (gemm3, index_u, index_a, index_b, index_v, m, n, k, l, ld_m, ld_a, ld_b, ld_c, false, false, true);
    delete index_u; delete index_v;
    return op;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_low_rank <T> *A, const h_index *index_a, const bool A_T, const dev_dense <T> *B, const h_index *index_b, const bool B_T) const
  {
    const int x_m = getNx(), y_m = getNy(), r_m = getRank(), x_a = A -> getNx(), y_a = A -> getNy(), r_a = A -> getRank(), x_b = B -> getNx(), y_b = B -> getNy();
    if (sameRowBasis(A))
    {
      const int m = x_m > x_b ? x_b : x_m, n = r_m, k = x_a > y_b ? y_b : x_a;
      const int ld_m = getLd_VT(), ld_a = B -> getLd(), ld_b = A -> getLd_VT();
      const h_index * index_v = self -> child_VT(this), * index_va = index_a -> child_VT(A);
      h_ops_tree * op = new h_ops_tree (gemm, index_v, index_b, index_va, m, n, k, ld_m, ld_a, ld_b, true, false);
      delete index_v; delete index_va;
      return op;
    }
    else
    {
      const int m = y_m > y_a ? y_a : y_m, n = r_m, k = r_a, l = x_a > y_b ? y_b : x_a, o = x_m > x_b ? x_b : x_m;
      const int ld_m = getLd_UxS(), ld_a = A -> getLd_UxS(), ld_b = A -> getLd_VT(), ld_c = B -> getLd(), ld_d = getLd_VT();
      const h_index * index_u = self -> child_UxS(this), * index_v = self -> child_VT(this);
      const h_index * index_ua = index_a -> child_UxS(A), * index_va = index_a -> child_VT(A);
      h_ops_tree * op = new h_ops_tree (gemm4, index_u, index_ua, index_va, index_b, index_v, m, n, k, l, o, ld_m, ld_a, ld_b, ld_c, ld_d, false, true, false, true);
      delete index_u; delete index_v; delete index_ua; delete index_va;
      return op;
    }
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_hierarchical <T> *A, const h_index *index_a, const bool A_T, const dev_dense <T> *B, const h_index *index_b, const bool B_T) const
  {
    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_h_element <T> *A, const h_index *index_a, const bool A_T, const dev_dense <T> *B, const h_index *index_b, const bool B_T) const
  {
    const dev_hierarchical <T> *h_a = A -> getElementHierarchical();
    const dev_low_rank <T> *lr_a = A -> getElementLowRank();
    const dev_dense <T> *d_a = A -> getElementDense();

    if (d_a != nullptr)
    { return generateOps_GEMM (self, d_a, index_a, A_T, B, index_b, B_T); }
    if (lr_a != nullptr)
    { return generateOps_GEMM (self, lr_a, index_a, A_T, B, index_b, B_T); }
    if (h_a != nullptr)
    { return generateOps_GEMM (self, h_a, index_a, A_T, B, index_b, B_T); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_dense <T> *A, const h_index *index_a, const bool A_T, const dev_low_rank <T> *B, const h_index *index_b, const bool B_T) const
  {
    const int x_m = getNx(), y_m = getNy(), r_m = getRank(), x_a = A -> getNx(), y_a = A -> getNy(), x_b = B -> getNx(), y_b = B -> getNy(), r_b = B -> getRank();
    if (sameColBasis(B))
    {
      const int m = y_m > y_a ? y_a : y_m, n = r_m, k = x_a > y_b ? y_b : x_a;
      const int ld_m = getLd_UxS(), ld_a = A -> getLd(), ld_b = B -> getLd_UxS();
      const h_index * index_u = self -> child_UxS(this), * index_ub = index_b -> child_UxS(B);
      h_ops_tree * op = new h_ops_tree (gemm, index_u, index_a, index_ub, m, n, k, ld_m, ld_a, ld_b, false, false);
      delete index_u; delete index_ub;
      return op;
    }
    else
    {
      const int m = y_m > y_a ? y_a : y_m, n = r_m, k = x_a > y_b ? y_b : x_a, l = r_b, o = x_m > x_b ? x_b : x_m;
      const int ld_m = getLd_UxS(), ld_a = A -> getLd(), ld_b = B -> getLd_UxS(), ld_c = B -> getLd_VT(), ld_d = getLd_VT();
      const h_index * index_u = self -> child_UxS(this), * index_v = self -> child_VT(this);
      const h_index * index_ub = index_b -> child_UxS(B), * index_vb = index_b -> child_VT(B);
      h_ops_tree * op = new h_ops_tree (gemm4, index_u, index_a, index_ub, index_vb, index_v, m, n, k, l, o, ld_m, ld_a, ld_b, ld_c, ld_d, false, false, true, true);
      delete index_u; delete index_v; delete index_ub; delete index_vb;
      return op;
    }
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_low_rank <T> *A, const h_index *index_a, const bool A_T, const dev_low_rank <T> *B, const h_index *index_b, const bool B_T) const
  {
    const int x_m = getNx(), y_m = getNy(), r_m = getRank(), x_a = A -> getNx(), y_a = A -> getNy(), r_a = A -> getRank(), x_b = B -> getNx(), y_b = B -> getNy(), r_b = B -> getRank();
    if (sameColBasis(B))
    {
      const int m = y_m > y_a ? y_a : y_m, n = r_m, k = r_a, l = x_a > y_b ? y_b : x_a;
      const int ld_m = getLd_UxS(), ld_a = A -> getLd_UxS(), ld_b = A -> getLd_VT(), ld_c = B -> getLd_UxS();
      const h_index * index_u = self -> child_UxS(this), * index_ua = index_a -> child_UxS(A);
      const h_index * index_va = index_a -> child_VT(A), * index_ub = index_b -> child_UxS(B);
      h_ops_tree * op = new h_ops_tree (gemm3, index_u, index_ua, index_va, index_ub, m, n, k, l, ld_m, ld_a, ld_b, ld_c, false, true, false);
      delete index_u; delete index_ua; delete index_va; delete index_ub;
      return op;
    }
    else if (sameRowBasis(A))
    {
      const int m = x_m > x_b ? x_b : x_m, n = r_m, k = r_b, l = x_a > y_b ? y_b : x_a;
      const int ld_m = getLd_VT(), ld_a = B -> getLd_VT(), ld_b = B -> getLd_UxS(), ld_c = A -> getLd_VT();
      const h_index * index_v = self -> child_VT(this), * index_va = index_a -> child_VT(A);
      const h_index * index_ub = index_b -> child_UxS(B), * index_vb = index_b -> child_VT(B);
      h_ops_tree * op = new h_ops_tree (gemm3, index_v, index_vb, index_ub, index_va, m, n, k, l, ld_m, ld_a, ld_b, ld_c, false, true, false);
      delete index_v; delete index_va; delete index_ub; delete index_vb;
      return op;
    }
    else
    {
      const int m = y_m > y_a ? y_a : y_m, n = r_m, k = r_a, l = x_a > y_b ? y_b : x_a, o = r_b, p = x_m > x_b ? x_b : x_m;
      const int ld_m = getLd_UxS(), ld_a = A -> getLd_UxS(), ld_b = A -> getLd_VT(), ld_c = B -> getLd_UxS(), ld_d = B -> getLd_VT(), ld_e = getLd_VT();
      const h_index * index_u = self -> child_UxS(this), * index_v = self -> child_VT(this);
      const h_index * index_ua = index_a -> child_UxS(A), * index_va = index_a -> child_VT(A);
      const h_index * index_ub = index_b -> child_UxS(B), * index_vb = index_b -> child_VT(B);
      h_ops_tree * op = new h_ops_tree (gemm5, index_u, index_ua, index_va, index_ub, index_vb, index_v,
        m, n, k, l, o, p, ld_m, ld_a, ld_b, ld_c, ld_d, ld_e, false, true, false, true, true);
      delete index_u; delete index_v; delete index_ua; delete index_va; delete index_ub; delete index_vb;
      return op;
    }
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_hierarchical <T> *A, const h_index *index_a, const bool A_T, const dev_low_rank <T> *B, const h_index *index_b, const bool B_T) const
  {
    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_h_element <T> *A, const h_index *index_a, const bool A_T, const dev_low_rank <T> *B, const h_index *index_b, const bool B_T) const
  {
    const dev_hierarchical <T> *h_a = A -> getElementHierarchical();
    const dev_low_rank <T> *lr_a = A -> getElementLowRank();
    const dev_dense <T> *d_a = A -> getElementDense();

    if (d_a != nullptr)
    { return generateOps_GEMM (self, d_a, index_a, A_T, B, index_b, B_T); }
    if (lr_a != nullptr)
    { return generateOps_GEMM (self, lr_a, index_a, A_T, B, index_b, B_T); }
    if (h_a != nullptr)
    { return generateOps_GEMM (self, h_a, index_a, A_T, B, index_b, B_T); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_dense <T> *A, const h_index *index_a, const bool A_T, const dev_hierarchical <T> *B, const h_index *index_b, const bool B_T) const
  {
    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_low_rank <T> *A, const h_index *index_a, const bool A_T, const dev_hierarchical <T> *B, const h_index *index_b, const bool B_T) const
  {
    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_hierarchical <T> *A, const h_index *index_a, const bool A_T, const dev_hierarchical <T> *B, const h_index *index_b, const bool B_T) const
  {
    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_h_element <T> *A, const h_index *index_a, const bool A_T, const dev_hierarchical <T> *B, const h_index *index_b, const bool B_T) const
  {
    const dev_hierarchical <T> *h_a = A -> getElementHierarchical();
    const dev_low_rank <T> *lr_a = A -> getElementLowRank();
    const dev_dense <T> *d_a = A -> getElementDense();

    if (d_a != nullptr)
    { return generateOps_GEMM (self, d_a, index_a, A_T, B, index_b, B_T); }
    if (lr_a != nullptr)
    { return generateOps_GEMM (self, lr_a, index_a, A_T, B, index_b, B_T); }
    if (h_a != nullptr)
    { return generateOps_GEMM (self, h_a, index_a, A_T, B, index_b, B_T); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_dense <T> *A, const h_index *index_a, const bool A_T, const dev_h_element <T> *B, const h_index *index_b, const bool B_T) const
  {
    const dev_hierarchical <T> *h_b = B -> getElementHierarchical();
    const dev_low_rank <T> *lr_b = B -> getElementLowRank();
    const dev_dense <T> *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, A_T, d_b, index_b, B_T); }
    if (lr_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, A_T, lr_b, index_b, B_T); }
    if (h_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, A_T, h_b, index_b, B_T); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_low_rank <T> *A, const h_index *index_a, const bool A_T, const dev_h_element <T> *B, const h_index *index_b, const bool B_T) const
  {
    const dev_hierarchical <T> *h_b = B -> getElementHierarchical();
    const dev_low_rank <T> *lr_b = B -> getElementLowRank();
    const dev_dense <T> *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, A_T, d_b, index_b, B_T); }
    if (lr_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, A_T, lr_b, index_b, B_T); }
    if (h_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, A_T, h_b, index_b, B_T); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_hierarchical <T> *A, const h_index *index_a, const bool A_T, const dev_h_element <T> *B, const h_index *index_b, const bool B_T) const
  {
    const dev_hierarchical <T> *h_b = B -> getElementHierarchical();
    const dev_low_rank <T> *lr_b = B -> getElementLowRank();
    const dev_dense <T> *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, A_T, d_b, index_b, B_T); }
    if (lr_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, A_T, lr_b, index_b, B_T); }
    if (h_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, A_T, h_b, index_b, B_T); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_h_element <T> *A, const h_index *index_a, const bool A_T, const dev_h_element <T> *B, const h_index *index_b, const bool B_T) const
  {
    const dev_hierarchical <T> *h_b = B -> getElementHierarchical();
    const dev_low_rank <T> *lr_b = B -> getElementLowRank();
    const dev_dense <T> *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, A_T, d_b, index_b, B_T); }
    if (lr_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, A_T, lr_b, index_b, B_T); }
    if (h_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, A_T, h_b, index_b, B_T); }

    return nullptr;
  }



  __host__ dev_dense <T> * convertToDense() const
  {
    dev_dense<T> * t1 = VT -> transpose();
    dev_dense<T> * t2 = UxS -> matrixMultiplication(t1);
    delete t1;
    return t2;
  }


  __host__ void print() const
  {
    printf("\n-- Low Rank: %d x %d, rank %d --\n", nx, ny, rank);
    UxS -> print();
    VT -> print();
  }

  __host__ void loadTestMatrix(const int x_start = 0, const int y_start = 0) const
  {
    UxS -> loadTestMatrix (x_start, y_start);
  }

};


#endif