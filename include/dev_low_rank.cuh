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

  __host__ dev_low_rank (const int x, const int y, const int rank_in = 0, dev_dense <T> * U_in = nullptr, dev_dense <T> * VT_in = nullptr)
  {
    nx = x;
    ny = y;

    cudaMallocManaged(&rank, sizeof(int), cudaMemAttachGlobal);
    * rank = (rank_in > 0 && rank_in <= x && rank_in <= y) ? rank_in : (x > y ? y : x);

    UxS = (U_in == nullptr) ? new dev_dense <T> (nx, ny) : U_in;
    VT = (VT_in == nullptr) ? new dev_dense <T> (nx, nx) : VT_in;
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

  __host__ T getElement (const int y, const int x) const
  {
    T element = 0;
    const int ld_u = UxS -> getLd(), ld_vt = VT -> getLd();
    const T * UxS_E = UxS -> getElements(), * VT_E = VT -> getElements();
    for (int i = 0; i < * rank; i++)
    { element += UxS_E[y * ld_u + i] * VT_E[x * ld_vt + i]; }
    return element;
  }

  __host__ dev_low_rank <T> ** createPartitions (const int y = 1, const int * ys = nullptr, const int x = 1, const int * xs = nullptr) const
  {
    if ((x > 1 && y > 0) || (y > 1 && x > 0)) 
    { 
      dev_low_rank <T> ** list = new dev_low_rank <T> * [x * y];
      const dev_dense <T> ** U_list = UxS -> createPartitions (y, ys, 1, nullptr);

      for (int i = 0; i < y; i++)
      {
        const int ny_i = ys[i + 1] - ys[i];
        const dev_dense <T> ** V_list = VT -> createPartitions (x, xs, 1, nullptr);

        for (int j = 0; j < x; j++)
        {
          const int nx_i = xs[j + 1] - xs[j];
          const dev_dense <T> ** U_dup = U_list[i] -> createPartitions (1, nullptr, 1, nullptr);
          list[i * x + j] = new dev_low_rank <T> (nx_i, ny_i, *rank, U_dup[0], V_list[j]);
          delete[] U_dup;
        }

        delete[] V_list;
      }

      delete[] U_list;
      return list;
    }
    else
    { 
      const dev_dense <T> ** U_dup = UxS -> createPartitions (1, nullptr, 1, nullptr), ** V_dup = VT -> createPartitions (1, nullptr, 1, nullptr);
      dev_low_rank <T> * ptr = new dev_low_rank <T> (nx, ny, *rank, U_dup[0], V_dup[0]);
      delete[] U_dup; delete[] V_dup;
      return new dev_low_rank <T> *[1] { ptr };
    }
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
    op -> resizeChildren(A -> getNx_blocks() * A -> getNy_blocks());

    int * y, * k, x = B -> getNx();
    x = (nx > x) ? x : nx;
    A -> getOffsets_y(&y);
    A -> getOffsets_x(&k);

#pragma omp parallel for num_threads(2)
    for (int i = 0; i < A -> getNx_blocks() * A -> getNy_blocks(); i++)
    {
      const int row = i / (A -> getNx_blocks()), col = i - row * (A -> getNx_blocks());
      const h_index index_ai = h_index (A, index_a, row, col), index_m = h_index (self, y[row], 0, index_ai.getNy(), x), index_bj = h_index (index_b, k[col], 0, index_ai.getNx(), x);
      h_ops_tree * op_i = generateOps_GEMM(&index_m, A -> getElement_blocks(row, col), &index_ai, B, &index_bj);
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
    op -> resizeChildren(A -> getNx_blocks() * A -> getNy_blocks());

    int * y, * k, x = B -> getNx();
    x = (nx > x) ? x : nx;
    A -> getOffsets_y(&y);
    A -> getOffsets_x(&k);

#pragma omp parallel for num_threads(2)
    for (int i = 0; i < A -> getNx_blocks() * A -> getNy_blocks(); i++)
    {
      const int row = i / (A -> getNx_blocks()), col = i - row * (A -> getNx_blocks());
      const h_index index_ai = h_index (A, index_a, row, col), index_m = h_index (self, y[row], 0, index_ai.getNy(), x), index_bj = h_index (index_b, k[col], 0, index_ai.getNx(), x);
      h_ops_tree * op_i = generateOps_GEMM(&index_m, A -> getElement_blocks(row, col), &index_ai, B, &index_bj);
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
    op -> resizeChildren(B -> getNx_blocks() * B -> getNy_blocks());

    int * x, * k, y = A -> getNy();
    y = (ny > y) ? y : ny;
    B -> getOffsets_y(&k);
    B -> getOffsets_x(&x);

#pragma omp parallel for num_threads(2)
    for (int i = 0; i < B -> getNx_blocks() * B -> getNy_blocks(); i++)
    {
      const int row = i / (B -> getNx_blocks()), col = i - row * (B -> getNx_blocks());
      const h_index index_bj = h_index (B, index_b, row, col), index_m = h_index (self, 0, x[col], y, index_bj.getNx()), index_ai = h_index (index_a, 0, k[row], y, index_bj.getNy());
      h_ops_tree * op_i = generateOps_GEMM(&index_m, A, &index_ai, B -> getElement_blocks(row, col), &index_bj);
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
    op -> resizeChildren(B -> getNx_blocks() * B -> getNy_blocks());

    int * x, * k, y = A -> getNy();
    y = (ny > y) ? y : ny;
    B -> getOffsets_y(&k);
    B -> getOffsets_x(&x);

#pragma omp parallel for num_threads(2)
    for (int i = 0; i < B -> getNx_blocks() * B -> getNy_blocks(); i++)
    {
      const int row = i / (B -> getNx_blocks()), col = i - row * (B -> getNx_blocks());
      const h_index index_bj = h_index (B, index_b, row, col), index_m = h_index (self, 0, x[col], y, index_bj.getNx()), index_ai = h_index (index_a, 0, k[row], y, index_bj.getNy());
      h_ops_tree * op_i = generateOps_GEMM(&index_m, A, &index_ai, B -> getElement_blocks(row, col), &index_bj);
      op -> setChild(op_i, i);
      delete op_i;
    }

    delete[] x;
    delete[] k;
    return op;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_hierarchical <T> *A, const h_index *index_a, const dev_hierarchical <T> *B, const h_index *index_b) const
  {
    if (A -> getNx_blocks() != B -> getNy_blocks())
    { printf("Matrices are partitioned differently in LR.H-H GEMM.\n"); return nullptr; }

    h_ops_tree * op = new h_ops_tree (gemm_lr_d_d, self, index_a, index_b);
    op -> resizeChildren(A -> getNy_blocks() * A -> getNx_blocks() * B -> getNx_blocks());

    int * x, * y;
    A -> getOffsets_y(&y);
    B -> getOffsets_x(&x);

#pragma omp parallel for num_threads(2)
    for (int i = 0; i < B -> getNx_blocks() * A -> getNy_blocks(); i++)
    {
      const int row = i / (B -> getNx_blocks()), col = i - row * (B -> getNx_blocks());
      for (int k = 0; k < A -> getNx_blocks(); k++)
      {
        const h_index index_ai = h_index (A, index_a, row, k), index_bj = h_index (B, index_b, k, col), index_m = h_index (self, y[row], x[col], index_ai.getNy(), index_bj.getNx());
        h_ops_tree * op_k = generateOps_GEMM(&index_m, A -> getElement_blocks(row, k), &index_ai, B -> getElement_blocks(k, col), &index_bj);
        op -> setChild(op_k, i * (B -> getNx_blocks() * A -> getNy_blocks()) + k);
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
    printf("\n-- LR: %d x %d, rank %d --\n", nx, ny, *rank);
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