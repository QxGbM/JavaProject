#ifndef _DEV_LOW_RANK_CUH
#define _DEV_LOW_RANK_CUH

#include <pspl.cuh>

template <class T> class dev_low_rank 
{
private:
  int nx;
  int ny;
  int * rank;
  dev_dense <T> * UxS;
  dev_dense <T> * VT;

public:

  __host__ dev_low_rank (const int x, const int y)
  {
    nx = x;
    ny = y;

    cudaMallocManaged(&rank, sizeof(int), cudaMemAttachGlobal);
    * rank = (x > y) ? y : x;

    UxS = new dev_dense <T> (nx, ny);
    VT = new dev_dense <T> (nx, nx);
  }

  __host__ ~dev_low_rank ()
  {
    delete UxS;
    delete VT;
  }

  __host__ inline int getNx () const { return nx; }

  __host__ inline int getNy () const { return ny; }

  __host__ inline int * getRank () const { return rank; }

  __host__ inline dev_dense <T> * getUxS () const { return UxS; }

  __host__ inline dev_dense <T> * getVT () const { return VT; }

  __host__ inline T * getElements (const int offset = 0) const 
  { 
    return offset >= getOffset_VT() ? VT -> getElements (offset - getOffset_VT()) : UxS -> getElements(offset); 
  }

  __host__ T getElement (const int x, const int y) const
  {
    T element = 0;
    for (int i = 0; i < * rank; i++)
    { element += (UxS -> getElements())[x * UxS -> getLd() + i] * (VT -> getElements())[y * VT -> getLd() + i]; }
    return element;
  }

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
    h_ops_tree * op = new h_ops_tree (gemm_lr_d_d, self, index_a, index_b);
    op -> resizeChildren(A -> getX() * A -> getY());

    int * y = new int[A -> getY()], * k = new int[A -> getX()], x = B -> getNx();
    y[0] = 0; k[0] = 0; x = (nx > x) ? x : nx;

    for (int i = 1; i < A -> getY(); i++)
    { y[i] = y[i - 1] + A -> getBlock(0, i - 1) -> getNy(); }

    for (int i = 1; i < A -> getX(); i++)
    { k[i] = k[i - 1] + A -> getBlock(i - 1, 0) -> getNx(); }

#pragma omp parallel for num_threads(2)
    for (int i = 0; i < A -> getX() * A -> getY(); i++)
    {
      const int row = i / (A -> getX()), col = i - row * (A -> getX());
      const h_index index_ai = h_index (A, index_a, row, col), index_m = h_index (self, y[row], 0, index_ai.getNy(), x), index_bj = h_index (index_b, k[col], 0, index_ai.getNx(), x);
      h_ops_tree * op_i = generateOps_GEMM(&index_m, A -> getBlock(col, row), &index_ai, B, &index_bj);
      op -> setChild(op_i, i);
      delete op_i;
    }

    delete[] y;
    delete[] k;
    return op;
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
    h_ops_tree * op = new h_ops_tree (gemm_lr_d_lr, self, index_a, index_b);
    op -> resizeChildren(A -> getX() * A -> getY());

    int * y = new int[A -> getY()], * k = new int[A -> getX()], x = B -> getNx();
    y[0] = 0; k[0] = 0; x = (nx > x) ? x : nx;

    for (int i = 1; i < A -> getY(); i++)
    { y[i] = y[i - 1] + A -> getBlock(0, i - 1) -> getNy(); }

    for (int i = 1; i < A -> getX(); i++)
    { k[i] = k[i - 1] + A -> getBlock(i - 1, 0) -> getNx(); }

#pragma omp parallel for num_threads(2)
    for (int i = 0; i < A -> getX() * A -> getY(); i++)
    {
      const int row = i / (A -> getX()), col = i - row * (A -> getX());
      const h_index index_ai = h_index (A, index_a, row, col), index_m = h_index (self, y[row], 0, index_ai.getNy(), x), index_bj = h_index (index_b, k[col], 0, index_ai.getNx(), x);
      h_ops_tree * op_i = generateOps_GEMM(&index_m, A -> getBlock(col, row), &index_ai, B, &index_bj);
      op -> setChild(op_i, i);
      delete op_i;
    }

    delete[] y;
    delete[] k;
    return op;
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
    h_ops_tree * op = new h_ops_tree (gemm_lr_d_d, self, index_a, index_b);
    op -> resizeChildren(B -> getX() * B -> getY());

    int * x = new int[B -> getX()], * k = new int[B -> getY()], y = A -> getNy();
    x[0] = 0; k[0] = 0; y = (ny > y) ? y : ny;

    for (int i = 1; i < B -> getX(); i++)
    { x[i] = x[i - 1] + B -> getBlock(i - 1, 0) -> getNx(); }

    for (int i = 1; i < B -> getY(); i++)
    { k[i] = k[i - 1] + B -> getBlock(0, i - 1) -> getNy(); }

#pragma omp parallel for num_threads(2)
    for (int i = 0; i < B -> getX() * B -> getY(); i++)
    {
      const int row = i / (B -> getX()), col = i - row * (B -> getX());
      const h_index index_bj = h_index (B, index_b, row, col), index_m = h_index (self, 0, x[col], y, index_bj.getNx()), index_ai = h_index (index_a, 0, k[row], y, index_bj.getNy());
      h_ops_tree * op_i = generateOps_GEMM(&index_m, A, &index_ai, B -> getBlock(col, row), &index_bj);
      op -> setChild(op_i, i);
      delete op_i;
    }

    delete[] x;
    delete[] k;
    return op;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_low_rank <T> *A, const h_index *index_a, const dev_hierarchical <T> *B, const h_index *index_b) const
  {
    h_ops_tree * op = new h_ops_tree (gemm_lr_lr_d, self, index_a, index_b);
    op -> resizeChildren(B -> getX() * B -> getY());

    int * x = new int[B -> getX()], * k = new int[B -> getY()], y = A -> getNy();
    x[0] = 0; k[0] = 0; y = (ny > y) ? y : ny;

    for (int i = 1; i < B -> getX(); i++)
    { x[i] = x[i - 1] + B -> getBlock(i - 1, 0) -> getNx(); }

    for (int i = 1; i < B -> getY(); i++)
    { k[i] = k[i - 1] + B -> getBlock(0, i - 1) -> getNy(); }

#pragma omp parallel for num_threads(2)
    for (int i = 0; i < B -> getX() * B -> getY(); i++)
    {
      const int row = i / (B -> getX()), col = i - row * (B -> getX());
      const h_index index_bj = h_index (B, index_b, row, col), index_m = h_index (self, 0, x[col], y, index_bj.getNx()), index_ai = h_index (index_a, 0, k[row], y, index_bj.getNy());
      h_ops_tree * op_i = generateOps_GEMM(&index_m, A, &index_ai, B -> getBlock(col, row), &index_bj);
      op -> setChild(op_i, i);
      delete op_i;
    }

    delete[] x;
    delete[] k;
    return op;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_hierarchical <T> *A, const h_index *index_a, const dev_hierarchical <T> *B, const h_index *index_b) const
  {
    if (A -> getX() != B -> getY())
    { printf("Matrices are partitioned differently in LR.H-H GEMM.\n"); return nullptr; }

    h_ops_tree * op = new h_ops_tree (gemm_lr_d_d, self, index_a, index_b);
    op -> resizeChildren(A -> getY() * A -> getX() * B -> getX());

    int * x = new int[A -> getY()], * y = new int[A -> getX()];
    x[0] = 0; y[0] = 0;

    for (int i = 1; i < B -> getX(); i++)
    { x[i] = x[i - 1] + B -> getBlock(i - 1, 0) -> getNx(); }

    for (int i = 1; i < A -> getY(); i++)
    { y[i] = y[i - 1] + A -> getBlock(0, i - 1) -> getNy(); }

#pragma omp parallel for num_threads(2)
    for (int i = 0; i < B -> getX() * A -> getY(); i++)
    {
      const int row = i / (B -> getX()), col = i - row * (B -> getX());
      for (int k = 0; k < A -> getX(); k++)
      {
        const h_index index_ai = h_index (A, index_a, row, k), index_bj = h_index (B, index_b, k, col), index_m = h_index (self, y[row], x[col], index_ai.getNy(), index_bj.getNx());
        h_ops_tree * op_k = generateOps_GEMM(&index_m, A -> getBlock(col, row), &index_ai, B, &index_bj);
        op -> setChild(op_k, i * (B -> getX() * A -> getY()) + k);
        delete op_k;
      }
    }

    delete[] x;
    delete[] y;
    return op;
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
    VT -> loadIdentityMatrix();
  }

};


#endif