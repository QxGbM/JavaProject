
#ifndef _DEV_HIERARCHICAL_ELEMENT_CUH
#define _DEV_HIERARCHICAL_ELEMENT_CUH

#include <pspl.cuh>

template <class T> class dev_h_element 
{
private:

  void * element;
  element_t type;

public:
  
  __host__ dev_h_element (void *element_in = nullptr, const element_t type_in = empty)
  {
    element = element_in;
    type = type_in;
  }

  __host__ ~dev_h_element ()
  { 
    dev_dense <T> *d = getElementDense();
    dev_low_rank <T> *lr = getElementLowRank();
    dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr) { delete d; }
    if (lr != nullptr) { delete lr; }
    if (h != nullptr) { delete h; }
  }

  __host__ dev_dense <T> * getElementDense() const
  {
    return (type == dense) ? ((dev_dense <T> *) element) : nullptr;
  }

  __host__ dev_low_rank <T> * getElementLowRank() const
  {
    return (type == low_rank) ? ((dev_low_rank <T> *) element) : nullptr;
  }

  __host__ dev_hierarchical <T> * getElementHierarchical() const
  {
    return (type == hierarchical) ? ((dev_hierarchical <T> *) element) : nullptr;
  }

  __host__ element_t getType() const
  {
    return type;
  }

  __host__ int getNx() const
  {
    const dev_dense <T> *d = getElementDense();
    const dev_low_rank <T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> getNx(); }
    if (lr != nullptr)
    { return lr -> getNx(); }
    if (h != nullptr)
    { return h -> getNx(); }

    return 0;
  }

  __host__ int getNy() const
  {
    const dev_dense <T> *d = getElementDense();
    const dev_low_rank <T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> getNy(); }
    if (lr != nullptr)
    { return lr -> getNy(); }
    if (h != nullptr)
    { return h -> getNy(); }

    return 0;
  }

  __host__ int getLd() const
  {
    const dev_dense <T> *d = getElementDense();

    if (d != nullptr)
    { return d -> getLd(); }

    return 0;
  }

  __host__ int getRank() const
  {
    const dev_low_rank <T> *lr = getElementLowRank();

    if (lr != nullptr)
    { return lr -> getRank(); }

    return 0;
  }

  __host__ T getElement (const int x_in, const int y_in) const
  {
    const dev_dense <T> *d = getElementDense();
    const dev_low_rank <T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return (d -> getElements(y_in * (d -> getLd()) + x_in))[0]; }
    if (lr != nullptr)
    { return lr -> getElement(x_in, y_in); }
    if (h != nullptr)
    { return h -> getElement(x_in, y_in); }

    return 0;
  }

  __host__ dev_dense <T> * convertToDense() const
  {
    const dev_dense <T> *d = getElementDense();
    const dev_low_rank <T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return new dev_dense <T> (d -> getNx(), d -> getNy(), d -> getElements(), d -> getLd()); }
    if (lr != nullptr)
    { return lr -> convertToDense(); }
    if (h != nullptr)
    { return h -> convertToDense(); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GETRF (const h_index *self) const
  {
    const dev_low_rank<T> *lr = getElementLowRank();
    if (lr != nullptr)
    { printf("A low-rank block cannot be LU decomposed.\n"); return nullptr; }

    h_ops_tree * ops = new h_ops_tree(getrf, self, getNx(), getNy(), getLd());
    const dev_hierarchical <T> *h = getElementHierarchical();
    if (h != nullptr) 
    { ops -> hookup_child (h -> generateOps_GETRF(self)); }
    return ops;
  }

  __host__ h_ops_tree * generateOps_TRSML (const h_index *self, const dev_h_element <T> *B, const h_index *index_b) const
  {
    if (B -> getNy() != getNy()) 
    { printf("Unmatched Dimension for TRSML.\n"); return nullptr; }

    const dev_low_rank<T> *lr = getElementLowRank(), *lr_b = B -> getElementLowRank();
    if (lr != nullptr)
    { printf("A low-rank block cannot be used for lower triangular solve.\n"); return nullptr; }

    const dev_hierarchical <T> *h = getElementHierarchical(), *h_b = B -> getElementHierarchical();
    const dev_dense<T> *d = getElementDense(), *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_TRSML (self, d_b, index_b); }
    if (lr_b != nullptr)
    { return generateOps_TRSML (self, lr_b, index_b); }
    if (h_b != nullptr)
    { return generateOps_TRSML (self, h_b, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_TRSML (const h_index *self, const dev_dense <T> *B, const h_index *index_b) const
  {
    h_ops_tree * ops = new h_ops_tree (trsml, index_b, self, B -> getNx(), getNy(), getNx(), B -> getLd(), getLd());
    const dev_hierarchical <T> *h = getElementHierarchical();
    if (h != nullptr)
    { ops -> hookup_child (h -> generateOps_TRSML (self, B, index_b)); }
    return ops;
  }

  __host__ h_ops_tree * generateOps_TRSML (const h_index *self, const dev_low_rank <T> *B, const h_index *index_b) const
  {
    const h_index * index_us_b = index_b -> child_UxS(B);
    h_ops_tree * ops = new h_ops_tree (trsml_lr, index_us_b, self, B -> getRank(), getNy(), getNx(), B -> getLd_UxS(), getLd(), false);
    delete index_us_b;

    const dev_hierarchical <T> *h = getElementHierarchical();
    if (h != nullptr)
    { ops -> hookup_child (h -> generateOps_TRSML (self, B, index_b)); }
    return ops;
  }

  __host__ h_ops_tree * generateOps_TRSML (const h_index *self, const dev_hierarchical <T> *B, const h_index *index_b) const
  {
    h_ops_tree * ops = new h_ops_tree (trsml, index_b, self, B -> getNx(), getNy(), getNx(), 0, getLd());
    const dev_dense <T> *d = getElementDense();
    const dev_hierarchical <T> *h = getElementHierarchical();
    if (d != nullptr)
    { ops -> hookup_child (B -> generateOps_TRSML_B (index_b, d, self)); }
    if (h != nullptr)
    { ops -> hookup_child (h -> generateOps_TRSML (self, B, index_b)); }
    return ops;
  }

  __host__ h_ops_tree * generateOps_TRSML_B (const h_index *self, const dev_dense <T> *B, const h_index *index_b) const
  {
    const dev_low_rank<T> *lr = getElementLowRank();
    if (lr != nullptr)
    { return new h_ops_tree (trsml_lr, self, index_b, lr -> getRank(), getNy(), B -> getNx(), lr -> getLd_UxS(), B -> getLd(), false); }

    h_ops_tree * ops = new h_ops_tree (trsml, self, index_b, getNx(), getNy(), B -> getNx(), getLd(), B -> getLd());
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (h != nullptr)
    { ops -> hookup_child(h -> generateOps_TRSML_B (self, B, index_b)); }

    return ops;
  }

  __host__ h_ops_tree * generateOps_TRSMR (const h_index *self, const dev_h_element <T> *B, const h_index *index_b) const
  {
    if (B -> getNx() != getNx()) 
    { printf("Unmatched Dimension for TRSMR.\n"); return nullptr; }

    const dev_hierarchical <T> *h_b = B -> getElementHierarchical();
    const dev_low_rank <T> *lr_b = B -> getElementLowRank();
    const dev_dense <T> *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_TRSMR (self, d_b, index_b); }
    if (lr_b != nullptr)
    { return generateOps_TRSMR (self, lr_b, index_b); }
    if (h_b != nullptr)
    { return generateOps_TRSMR (self, h_b, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_TRSMR (const h_index *self, const dev_dense <T> *B, const h_index *index_b) const
  {
    h_ops_tree * ops = new h_ops_tree (trsmr, index_b, self, getNx(), B -> getNy(), getNy(), B -> getLd(), getLd());
    const dev_hierarchical <T> *h = getElementHierarchical();
    if (h != nullptr)
    { ops -> hookup_child(h -> generateOps_TRSMR(self, B, index_b)); }
    return ops;
  }

  __host__ h_ops_tree * generateOps_TRSMR (const h_index *self, const dev_low_rank <T> *B, const h_index *index_b) const
  {
    const h_index * index_vt_b = index_b -> child_VT(B);
    h_ops_tree * ops = new h_ops_tree (trsmr_lr, index_vt_b, self, getNx(), B -> getRank(), getNy(), B -> getLd_VT(), getLd(), true);
    delete index_vt_b;

    const dev_hierarchical <T> *h = getElementHierarchical();
    if (h != nullptr)
    { ops -> hookup_child(h -> generateOps_TRSMR(self, B, index_b)); }
    return ops;
  }

  __host__ h_ops_tree * generateOps_TRSMR (const h_index *self, const dev_hierarchical <T> *B, const h_index *index_b) const
  {
    h_ops_tree * ops = new h_ops_tree(trsmr, index_b, self, getNx(), B -> getNy(), getNy(), 0, getLd());
    const dev_hierarchical <T> *h = getElementHierarchical();
    if (h != nullptr)
    { ops -> hookup_child(h -> generateOps_TRSMR(self, B, index_b)); }
    return ops;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_h_element <T> *A, const h_index *index_a, const bool A_T, const dev_h_element <T> *B, const h_index *index_b, const bool B_T) const
  {
    if ((A -> getNy() != getNy()) || (B -> getNx() != getNx()) || (A -> getNx() != B -> getNy())) 
    { printf("Unmatched Dimension for GEMM.\n"); return nullptr; }

    const dev_hierarchical<T> *h_b = B -> getElementHierarchical();
    const dev_dense<T> *d_b = B -> getElementDense();
    const dev_low_rank<T> *lr_b = B -> getElementLowRank();

    if (d_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, A_T, d_b, index_b, B_T); }
    if (lr_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, A_T, lr_b, index_b, B_T); }
    if (h_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, A_T, h_b, index_b, B_T); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_dense <T> *A, const h_index *index_a, const bool A_T, const dev_h_element <T> *B, const h_index *index_b, const bool B_T) const
  {
    if (B -> getNx() != getNx())
    { printf("Unmatched Dimension for GEMM.\n"); return nullptr; }

    const dev_hierarchical<T> *h_b = B -> getElementHierarchical();
    const dev_dense<T> *d_b = B -> getElementDense();
    const dev_low_rank<T> *lr_b = B -> getElementLowRank();

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
    if (B -> getNx() != getNx())
    { printf("Unmatched Dimension for GEMM.\n"); return nullptr; }

    const dev_hierarchical<T> *h_b = B -> getElementHierarchical();
    const dev_dense<T> *d_b = B -> getElementDense();
    const dev_low_rank<T> *lr_b = B -> getElementLowRank();

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
    if (B -> getNx() != getNx())
    { printf("Unmatched Dimension for GEMM.\n"); return nullptr; }

    const dev_hierarchical<T> *h_b = B -> getElementHierarchical();
    const dev_dense<T> *d_b = B -> getElementDense();
    const dev_low_rank<T> *lr_b = B -> getElementLowRank();

    if (d_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, A_T, d_b, index_b, B_T); }
    if (lr_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, A_T, lr_b, index_b, B_T); }
    if (h_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, A_T, h_b, index_b, B_T); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_h_element <T> *A, const h_index *index_a, const bool A_T, const dev_dense <T> *B, const h_index *index_b, const bool B_T) const
  {
    if (A -> getNy() != getNy())
    { printf("Unmatched Dimension for GEMM.\n"); return nullptr; }

    const dev_hierarchical<T> *h_a = A -> getElementHierarchical();
    const dev_dense<T> *d_a = A -> getElementDense();
    const dev_low_rank<T> *lr_a = A -> getElementLowRank();

    if (d_a != nullptr)
    { return generateOps_GEMM (self, d_a, index_a, A_T, B, index_b, B_T); }
    if (lr_a != nullptr)
    { return generateOps_GEMM (self, lr_a, index_a, A_T, B, index_b, B_T); }
    if (h_a != nullptr)
    { return generateOps_GEMM (self, h_a, index_a, A_T, B, index_b, B_T); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_h_element <T> *A, const h_index *index_a, const bool A_T, const dev_low_rank <T> *B, const h_index *index_b, const bool B_T) const
  {
    if (A -> getNy() != getNy())
    { printf("Unmatched Dimension for GEMM.\n"); return nullptr; }

    const dev_hierarchical<T> *h_a = A -> getElementHierarchical();
    const dev_dense<T> *d_a = A -> getElementDense();
    const dev_low_rank<T> *lr_a = A -> getElementLowRank();

    if (d_a != nullptr)
    { return generateOps_GEMM (self, d_a, index_a, A_T, B, index_b, B_T); }
    if (lr_a != nullptr)
    { return generateOps_GEMM (self, lr_a, index_a, A_T, B, index_b, B_T); }
    if (h_a != nullptr)
    { return generateOps_GEMM (self, h_a, index_a, A_T, B, index_b, B_T); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_h_element <T> *A, const h_index *index_a, const bool A_T, const dev_hierarchical <T> *B, const h_index *index_b, const bool B_T) const
  {
    if (A -> getNy() != getNy())
    { printf("Unmatched Dimension for GEMM.\n"); return nullptr; }

    const dev_hierarchical<T> *h_a = A -> getElementHierarchical();
    const dev_dense<T> *d_a = A -> getElementDense();
    const dev_low_rank<T> *lr_a = A -> getElementLowRank();

    if (d_a != nullptr)
    { return generateOps_GEMM (self, d_a, index_a, A_T, B, index_b, B_T); }
    if (lr_a != nullptr)
    { return generateOps_GEMM (self, lr_a, index_a, A_T, B, index_b, B_T); }
    if (h_a != nullptr)
    { return generateOps_GEMM (self, h_a, index_a, A_T, B, index_b, B_T); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *index_m, const dev_dense <T> *A, const h_index *index_a, const bool A_T, const dev_dense <T> *B, const h_index *index_b, const bool B_T) const
  {
    const dev_low_rank  <T> *lr = getElementLowRank();
    if (lr != nullptr)
    {
      const h_index * index_vt = index_b -> child(-1, lr -> getOffset_VT());
      h_ops_tree * ops = new h_ops_tree (gemm3, index_m, index_a, index_b, index_vt, 
        getNy(), lr -> getRank(), A -> getNx(), getNx(), 
        lr -> getRank(), A -> getLd(), B -> getLd(), lr -> getRank(), A_T, B_T, false);
      delete index_vt;
      return ops;
    }

    h_ops_tree * ops = new h_ops_tree (gemm, index_m, index_a, index_b, 
      getNy(), getNx(), A -> getNx(), 
      getLd(), A -> getLd(), B -> getLd(), A_T, B_T);
    const dev_hierarchical <T> *h = getElementHierarchical();
    if (h != nullptr)
    { ops -> hookup_child (h -> generateOps_GEMM (index_m, A, index_a, A_T, B, index_b, B_T)); }
    return ops;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *index_m, const dev_dense <T> *A, const h_index *index_a, const bool A_T, const dev_low_rank <T> *B, const h_index *index_b, const bool B_T) const
  {
    const dev_low_rank <T> *lr = getElementLowRank();
    const h_index * index_vt_b = index_b -> child (-1, B -> getOffset_VT());
    if (lr != nullptr)
    {
      const h_index * index_vt = index_b -> child (-1, lr -> getOffset_VT());
      h_ops_tree * ops = new h_ops_tree (gemm4, index_m, index_a, index_b, index_vt_b, index_vt, 
        getNy(), lr -> getRank(), A -> getNx(), B -> getRank(), getNx(),
        lr -> getRank(), A -> getLd(), B -> getRank(), B -> getRank(), lr -> getRank(), A_T, B_T, !B_T, false);
      delete index_vt;
      return ops;
    }

    h_ops_tree * ops = new h_ops_tree (gemm3, index_m, index_a, index_b, index_vt_b, 
      getNy(), getNx(), A -> getNx(), B -> getRank(), 
      getLd(), A -> getLd(), B -> getRank(), B -> getRank(), A_T, B_T, !B_T);
    delete index_vt_b;
    const dev_hierarchical <T> *h = getElementHierarchical();
    if (h != nullptr)
    {  }
    return ops;
  }
  
  __host__ h_ops_tree * generateOps_GEMM (const h_index *index_m, const dev_dense <T> *A, const h_index *index_a, const bool A_T, const dev_hierarchical <T> *B, const h_index *index_b, const bool B_T) const
  {
    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *index_m, const dev_low_rank <T> *A, const h_index *index_a, const bool A_T, const dev_dense <T> *B, const h_index *index_b, const bool B_T) const
  {
    const dev_low_rank <T> *lr = getElementLowRank();
    const h_index * index_vt_a = index_a -> child (-1, A -> getOffset_VT());
    if (lr != nullptr)
    {
      const h_index * index_vt = index_b -> child (-1, lr -> getOffset_VT());
      h_ops_tree * ops = new h_ops_tree (gemm4, index_m, index_a, index_vt_a, index_b, index_vt, 
        getNy(), lr -> getRank(), A -> getRank(), A -> getNx(), getNx(),
        lr -> getRank(), A -> getRank(), A -> getRank(), B -> getLd(), lr -> getRank(), A_T, !A_T, B_T, false);
      delete index_vt;
      return ops;
    }

    h_ops_tree * ops = new h_ops_tree (gemm3, index_m, index_a, index_vt_a, index_b, 
      getNy(), getNx(), A -> getRank(), A -> getRank(),
      getLd(), A -> getRank(), A -> getRank(), B -> getLd(), A_T, !A_T, B_T);
    delete index_vt_a;
    const dev_hierarchical <T> *h = getElementHierarchical();
    if (h != nullptr)
    {  }
    return ops;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *index_m, const dev_low_rank <T> *A, const h_index *index_a, const bool A_T, const dev_low_rank <T> *B, const h_index *index_b, const bool B_T) const
  {
    const dev_low_rank <T> *lr = getElementLowRank();
	const h_index * index_us_a = index_a -> child_UxS(A), * index_vt_a = index_a -> child_VT(A);
    const h_index * index_us_b = index_b -> child_UxS(B), * index_vt_b = index_b -> child_VT(B);
	h_ops_tree * ops = nullptr;

    if (lr != nullptr)
    {
      const h_index * index_us_m = index_m -> child_UxS(lr), * index_vt_m = index_m -> child_VT (lr);
      ops = new h_ops_tree (gemm5, index_us_m, index_us_a, index_vt_a, index_us_b, index_vt_b, index_vt_m, 
        getNy(), lr -> getRank(), A -> getRank(), A -> getNx(), B -> getRank(), getNx(),
        lr -> getRank(), A -> getRank(), A -> getRank(), B -> getRank(), B -> getRank(), lr -> getRank(), A_T, !A_T, B_T, !B_T, false);
	  delete index_us_m; delete index_vt_m;
    }
	else
	{
      ops = new h_ops_tree (gemm4, index_m, index_us_a, index_vt_a, index_us_b, index_vt_b, 
        getNy(), getNx(), A -> getRank(), A -> getNx(), B -> getRank(),
        getLd(), A -> getRank(), A -> getRank(), B -> getRank(), B -> getRank(), A_T, !A_T, B_T, !B_T);

      const dev_hierarchical <T> *h = getElementHierarchical();
      if (h != nullptr)
      { ops -> hookup_child (h -> generateOps_GEMM (index_m, A, index_a, A_T, B, index_b, B_T)); }
	}

	delete index_us_a; delete index_vt_a;
	delete index_us_b; delete index_vt_b;

	return ops;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *index_m, const dev_low_rank <T> *A, const h_index *index_a, const bool A_T, const dev_hierarchical <T> *B, const h_index *index_b, const bool B_T) const
  {
    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *index_m, const dev_hierarchical <T> *A, const h_index *index_a, const bool A_T, const dev_dense <T> *B, const h_index *index_b, const bool B_T) const
  {
    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *index_m, const dev_hierarchical <T> *A, const h_index *index_a, const bool A_T, const dev_low_rank <T> *B, const h_index *index_b, const bool B_T) const
  {
    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *index_m, const dev_hierarchical <T> *A, const h_index *index_a, const bool A_T, const dev_hierarchical <T> *B, const h_index *index_b, const bool B_T) const
  {
    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM_A (const dev_dense <T> *M, const h_index *index_m, const h_index *index_a, const bool A_T, const dev_dense <T> *B, const h_index *index_b, const bool B_T) const
  {
    h_ops_tree * ops = new h_ops_tree (gemm, index_m, index_a, index_b, getNy(), M -> getNx(), getNx(), M -> getLd(), getLd(), B -> getLd(), A_T, B_T);
    const dev_hierarchical <T> *h_a = getElementHierarchical();
    if (h_a != nullptr)
    { } //TODO
    return ops;
  }

  __host__ h_ops_tree * generateOps_GEMM_A (const dev_low_rank <T> *M, const h_index *index_m, const h_index *index_a, const bool A_T, const dev_low_rank <T> *B, const h_index *index_b, const bool B_T) const
  {
    const dev_low_rank <T> *lr_a = getElementLowRank();

    const h_index * index_us_m = index_m -> child_UxS(M), * index_vt_m = index_m -> child_VT(M);
    const h_index * index_us_b = index_b -> child_UxS(B), * index_vt_b = index_b -> child_VT(B);

    h_ops_tree * ops;

    if (lr_a != nullptr)
    {
      const h_index * index_us_a = index_a -> child_UxS(lr_a), *index_vt_a = index_a -> child_VT(lr_a);
      
      if (M == B)
      {
        ops = new h_ops_tree (gemm3, index_us_m, index_us_a, index_vt_a, index_us_b,
          getNy(), M -> getRank(), lr_a -> getRank(), getNx(),
          M -> getRank(), lr_a -> getRank(), lr_a -> getRank(), B -> getRank(), A_T, !A_T, B_T);
      }
      else
      {
        ops = new h_ops_tree (gemm5, index_us_m, index_us_a, index_vt_a, index_us_b, index_vt_b, index_vt_m,
          getNy(), M -> getRank(), lr_a -> getRank(), getNx(), B -> getRank(), B -> getNx(),
          M -> getRank(), lr_a -> getRank(), lr_a -> getRank(), B -> getRank(), B -> getRank(), M -> getRank(), A_T, !A_T, B_T, !B_T, false);
      }

      delete index_us_a; delete index_vt_a;
      return ops;
    }
    else
    {
      if (M == B)
      {
        ops = new h_ops_tree (gemm, index_us_m, index_a, index_us_b,
          getNy(), M -> getRank(), getNx(),
          M -> getRank(), getLd(), B -> getRank(), A_T, B_T);
      }
      else
      {
        ops = new h_ops_tree (gemm4, index_us_m, index_a, index_us_b, index_vt_b, index_vt_m, 
          getNy(), M -> getRank(), getNx(), B -> getRank(), B -> getNx(),
          M -> getRank(), getLd(), B -> getRank(), B -> getRank(), M -> getRank(), A_T, B_T, !B_T, false);
      }

      const dev_hierarchical <T> *h_a = getElementHierarchical();
      if (h_a != nullptr)
      {
        //TODO
      }
    }

    delete index_us_m; delete index_vt_m;
    delete index_us_b; delete index_vt_b;

    return ops;
  }

  __host__ h_ops_tree * generateOps_GEMM_B (const dev_dense <T> *M, const h_index *index_m, const dev_dense <T> *A, const h_index *index_a, const bool A_T, const h_index *index_b, const bool B_T) const
  {
    h_ops_tree * ops = new h_ops_tree(gemm, index_m, index_a, index_b, M -> getNy(), getNx(), getNy(), M -> getLd(), A -> getLd(), getLd(), A_T, B_T);
    const dev_hierarchical <T> *h_b = getElementHierarchical();
    if (h_b != nullptr)
    { } //TODO
    return ops;
  }

  __host__ h_ops_tree * generateOps_GEMM_B (const dev_low_rank <T> *M, const h_index *index_m, const dev_low_rank <T> *A, const h_index *index_a, const bool A_T, const h_index *index_b, const bool B_T) const
  {
    const dev_low_rank <T> *lr_b = getElementLowRank();

    const h_index * index_us_m = index_m -> child_UxS(M), * index_vt_m = index_m -> child_VT(M);
    const h_index * index_us_a = index_a -> child_UxS(A), * index_vt_a = index_a -> child_VT(A);

    h_ops_tree * ops;

    if (lr_b != nullptr)
    {
      const h_index * index_us_b = index_b -> child_UxS(lr_b), *index_vt_b = index_b -> child_VT(lr_b);
      
      if (M == A)
      {
        ops = new h_ops_tree (gemm3, index_vt_m, index_vt_b, index_us_b, index_vt_a,
          getNx(), M -> getRank(), lr_b -> getRank(), getNy(),
          M -> getRank(), lr_b -> getRank(), lr_b -> getRank(), A -> getRank(), B_T, !B_T, A_T);
      }
      else
      {
        ops = new h_ops_tree (gemm5, index_us_m, index_us_a, index_vt_a, index_us_b, index_vt_b, index_vt_m,
          getNy(), M -> getRank(), A -> getRank(), A -> getNx(), lr_b -> getRank(), getNx(),
          M -> getRank(), A -> getRank(), A -> getRank(), lr_b -> getRank(), lr_b -> getRank(), M -> getRank(), A_T, !A_T, B_T, !B_T, false);
      }

      delete index_us_b; delete index_vt_b;
      return ops;
    }
    else
    {
      if (M == A)
      {
        ops = new h_ops_tree (gemm, index_vt_m, index_b, index_vt_a,
          getNx(), M -> getRank(), getNy(),
          M -> getRank(), getLd(), A -> getRank(), !B_T, A_T);
      }
      else
      {
        ops = new h_ops_tree (gemm4, index_us_m, index_us_a, index_vt_a, index_b, index_vt_m, 
          getNy(), M -> getRank(), A -> getRank(), A -> getNx(), getNx(),
          M -> getRank(), A -> getRank(), A -> getRank(), getLd(), M -> getRank(), A_T, !A_T, B_T, false);
      }

      const dev_hierarchical <T> *h_b = getElementHierarchical();
      if (h_b != nullptr)
      {
        //TODO
      }
    }

    delete index_us_m; delete index_vt_m;
    delete index_us_a; delete index_vt_a;

    return ops;
  }

  __host__ void print (const h_index *index) const
  {

    const dev_dense <T> *d = getElementDense();
    const dev_low_rank <T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    index -> print();

    if (d != nullptr) { d -> print(); }
    if (lr != nullptr) { lr -> print(); }
    if (h != nullptr) { h -> print(index); } 
  }

};


#endif