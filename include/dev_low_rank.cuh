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
  { 
    return lr -> UxS == UxS;
  }

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

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_dense <T> *A, const h_index *index_a, const dev_dense <T> *B, const h_index *index_b) const
  {
    return new h_ops_tree (gemm_lr_d_d, self, index_a, index_b);
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_low_rank <T> *A, const h_index *index_a, const dev_dense <T> *B, const h_index *index_b) const
  {
    return new h_ops_tree (gemm_lr_lr_d, self, index_a, index_b);
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_hierarchical <T> *A, const h_index *index_a, const dev_dense <T> *B, const h_index *index_b) const
  {
    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_h_element <T> *A, const h_index *index_a, const dev_dense <T> *B, const h_index *index_b) const
  {
    const dev_hierarchical <T> *h_a = A -> getElementHierarchical();
    const dev_low_rank <T> *lr_a = A -> getElementLowRank();
    const dev_dense <T> *d_a = A -> getElementDense();

    if (d_a != nullptr)
    { return generateOps_GEMM (self, d_a, index_a, B, index_b); }
    if (lr_a != nullptr)
    { return generateOps_GEMM (self, lr_a, index_a, B, index_b); }
    if (h_a != nullptr)
    { return generateOps_GEMM (self, h_a, index_a, B, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_dense <T> *A, const h_index *index_a, const dev_low_rank <T> *B, const h_index *index_b) const
  {
    return new h_ops_tree (gemm_lr_d_lr, self, index_a, index_b);
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_low_rank <T> *A, const h_index *index_a, const dev_low_rank <T> *B, const h_index *index_b) const
  {
    return new h_ops_tree (gemm_lr_lr_lr, self, index_a, index_b);
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_hierarchical <T> *A, const h_index *index_a, const dev_low_rank <T> *B, const h_index *index_b) const
  {
    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_h_element <T> *A, const h_index *index_a, const dev_low_rank <T> *B, const h_index *index_b) const
  {
    const dev_hierarchical <T> *h_a = A -> getElementHierarchical();
    const dev_low_rank <T> *lr_a = A -> getElementLowRank();
    const dev_dense <T> *d_a = A -> getElementDense();

    if (d_a != nullptr)
    { return generateOps_GEMM (self, d_a, index_a, B, index_b); }
    if (lr_a != nullptr)
    { return generateOps_GEMM (self, lr_a, index_a, B, index_b); }
    if (h_a != nullptr)
    { return generateOps_GEMM (self, h_a, index_a, B, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_dense <T> *A, const h_index *index_a, const dev_hierarchical <T> *B, const h_index *index_b) const
  {
    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_low_rank <T> *A, const h_index *index_a, const dev_hierarchical <T> *B, const h_index *index_b) const
  {
    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_hierarchical <T> *A, const h_index *index_a, const dev_hierarchical <T> *B, const h_index *index_b) const
  {
    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_h_element <T> *A, const h_index *index_a, const dev_hierarchical <T> *B, const h_index *index_b) const
  {
    const dev_hierarchical <T> *h_a = A -> getElementHierarchical();
    const dev_low_rank <T> *lr_a = A -> getElementLowRank();
    const dev_dense <T> *d_a = A -> getElementDense();

    if (d_a != nullptr)
    { return generateOps_GEMM (self, d_a, index_a, B, index_b); }
    if (lr_a != nullptr)
    { return generateOps_GEMM (self, lr_a, index_a, B, index_b); }
    if (h_a != nullptr)
    { return generateOps_GEMM (self, h_a, index_a, B, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_dense <T> *A, const h_index *index_a, const dev_h_element <T> *B, const h_index *index_b) const
  {
    const dev_hierarchical <T> *h_b = B -> getElementHierarchical();
    const dev_low_rank <T> *lr_b = B -> getElementLowRank();
    const dev_dense <T> *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, d_b, index_b); }
    if (lr_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, lr_b, index_b); }
    if (h_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, h_b, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_low_rank <T> *A, const h_index *index_a, const dev_h_element <T> *B, const h_index *index_b) const
  {
    const dev_hierarchical <T> *h_b = B -> getElementHierarchical();
    const dev_low_rank <T> *lr_b = B -> getElementLowRank();
    const dev_dense <T> *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, d_b, index_b); }
    if (lr_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, lr_b, index_b); }
    if (h_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, h_b, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_hierarchical <T> *A, const h_index *index_a, const dev_h_element <T> *B, const h_index *index_b) const
  {
    const dev_hierarchical <T> *h_b = B -> getElementHierarchical();
    const dev_low_rank <T> *lr_b = B -> getElementLowRank();
    const dev_dense <T> *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, d_b, index_b); }
    if (lr_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, lr_b, index_b); }
    if (h_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, h_b, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_h_element <T> *A, const h_index *index_a, const dev_h_element <T> *B, const h_index *index_b) const
  {
    const dev_hierarchical <T> *h_b = B -> getElementHierarchical();
    const dev_low_rank <T> *lr_b = B -> getElementLowRank();
    const dev_dense <T> *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, d_b, index_b); }
    if (lr_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, lr_b, index_b); }
    if (h_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, h_b, index_b); }

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
    VT -> loadRandomOrthMatrix(x_start);
  }

};


#endif