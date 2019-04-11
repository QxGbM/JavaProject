
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
    { return lr -> getNy_VT(); }
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
    { return lr -> getNy_U(); }
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

  __host__ T getElement(const int x_in, const int y_in) const
  {
    const dev_dense <T> *d = getElementDense();
    const dev_low_rank <T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    {
      return (d -> getElements(y_in * (d -> getLd()) + x_in))[0];
    }
    if (lr != nullptr)
    {
      
    }
    if (h != nullptr)
    {
      return h -> getElement(x_in, y_in);
    }

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
    h_ops_tree * ops = new h_ops_tree(trsml, index_b, self, B -> getNx(), getNy(), getNx(), B -> getLd(), getLd());
    if (h != nullptr) 
    { 
      if (h_b != nullptr)
      { ops -> hookup_child (h -> generateOps_TRSML(self, h_b, index_b)); }
      if (d_b != nullptr)
      { ops -> hookup_child (h -> generateOps_TRSML(self, d_b, index_b)); }
      if (lr_b != nullptr)
      {  }
    }
    if (d != nullptr)
    {
      if (h_b != nullptr)
      { ops -> hookup_child (h_b -> generateOps_TRSML_B(index_b, d, self)); }
      if (lr_b != nullptr)
      {
        
      }
    }
    return ops;
  }

  __host__ h_ops_tree * generateOps_TRSML (const h_index *self, const dev_dense <T> *B, const h_index *index_b) const
  {
    h_ops_tree * ops = new h_ops_tree(trsml, index_b, self, B -> getNx(), getNy(), getNx(), B -> getLd(), getLd());
    const dev_hierarchical <T> *h = getElementHierarchical();
    if (h != nullptr)
    { ops -> hookup_child(h -> generateOps_TRSML (self, B, index_b)); }
    return ops;
  }

  __host__ h_ops_tree * generateOps_TRSML_B (const h_index *self, const dev_dense <T> *B, const h_index *index_b) const
  {
    h_ops_tree * ops = new h_ops_tree(trsml, self, index_b, getNx(), getNy(), B -> getNx(), getLd(), B -> getLd());
    const dev_hierarchical <T> *h = getElementHierarchical();
    const dev_low_rank<T> *lr = getElementLowRank();
    if (h != nullptr)
    { ops -> hookup_child(h -> generateOps_TRSML_B (self, B, index_b)); }
    if (lr != nullptr)
    {

    }
    return ops;
  }

  __host__ h_ops_tree * generateOps_TRSMR (const h_index *self, const dev_h_element <T> *B, const h_index *index_b) const
  {
    if (B -> getNx() != getNx()) 
    { printf("Unmatched Dimension for TRSMR.\n"); return nullptr; }

    h_ops_tree * ops = new h_ops_tree(trsmr, index_b, self, getNx(), B -> getNy(), getNy(), B -> getLd(), getLd());
    const dev_hierarchical <T> *h = getElementHierarchical(), *h_b = B -> getElementHierarchical();
    const dev_dense <T> *d_b = B -> getElementDense();
    if (h != nullptr) 
    {
      if (h_b != nullptr)
      { ops -> hookup_child(h -> generateOps_TRSMR(self, h_b, index_b)); }
      if (d_b != nullptr)
      { ops -> hookup_child(h -> generateOps_TRSMR(self, d_b, index_b)); }
    }
    return ops;
  }

  __host__ h_ops_tree * generateOps_TRSMR (const h_index *self, const dev_dense <T> *B, const h_index *index_b) const
  {
    h_ops_tree * ops = new h_ops_tree(trsmr, index_b, self, getNx(), B -> getNy(), getNy(), B -> getLd(), getLd());
    const dev_hierarchical <T> *h = getElementHierarchical();
    if (h != nullptr)
    { ops -> hookup_child(h -> generateOps_TRSMR(self, B, index_b)); }
    return ops;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_h_element <T> *A, const h_index *index_a, const bool A_T, const dev_h_element <T> *B, const h_index *index_b, const bool B_T) const
  {
    if ((A -> getNy() != getNy()) || (B -> getNx() != getNx()) || (A -> getNx() != B -> getNy())) 
    { printf("Unmatched Dimension for GEMM.\n"); return nullptr; }

    h_ops_tree * ops = new h_ops_tree(gemm, self, index_a, index_b, getNy(), getNx(), A -> getNx(), getLd(), A -> getLd(), B -> getLd(), A_T, B_T);
    const dev_hierarchical <T> *h = getElementHierarchical(), *h_a = A -> getElementHierarchical(), *h_b = B -> getElementHierarchical();
    const dev_dense <T> *d = getElementDense(), *d_a = A -> getElementDense(), *d_b = B -> getElementDense();
    if (h != nullptr)
    {
      if (d_a != nullptr && d_b != nullptr)
      { ops -> hookup_child(h -> generateOps_GEMM (self, d_a, index_a, A_T, d_b, index_b, B_T)); }
      // TODO
    }
    return ops;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_dense <T> *A, const h_index *index_a, const bool A_T, const dev_h_element <T> *B, const h_index *index_b, const bool B_T) const
  {
    if ((A -> getNy() != getNy()) || (B -> getNx() != getNx()) || (A -> getNx() != B -> getNy())) 
    { printf("Unmatched Dimension for GEMM.\n"); return nullptr; }

    h_ops_tree * ops = new h_ops_tree(gemm, self, index_a, index_b, getNy(), getNx(), A -> getNx(), getLd(), A -> getLd(), B -> getLd(), A_T, B_T);
    const dev_hierarchical <T> *h = getElementHierarchical(), *h_b = B -> getElementHierarchical();
    const dev_dense <T> *d = getElementDense(), *d_b = B -> getElementDense();
    if (h != nullptr)
    {
      if (d_b != nullptr)
      { ops -> hookup_child(h -> generateOps_GEMM (self, A, index_a, A_T, d_b, index_b, B_T)); }
      // TODO
    }
    return ops;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *index_m, const dev_dense <T> *A, const h_index *index_a, const bool A_T, const dev_dense <T> *B, const h_index *index_b, const bool B_T) const
  {
    h_ops_tree * ops = new h_ops_tree(gemm, index_m, index_a, index_b, getNy(), getNx(), A -> getNx(), getLd(), A -> getLd(), B -> getLd(), A_T, B_T);
    const dev_hierarchical <T> *h = getElementHierarchical();
    if (h != nullptr)
    { ops -> hookup_child(h -> generateOps_GEMM (index_m, A, index_a, A_T, B, index_b, B_T)); }
    return ops;
  }

  __host__ h_ops_tree * generateOps_GEMM_A (const dev_dense <T> *M, const h_index *index_m, const h_index *index_a, const bool A_T, const dev_dense <T> *B, const h_index *index_b, const bool B_T) const
  {
    h_ops_tree * ops = new h_ops_tree(gemm, index_m, index_a, index_b, getNy(), M -> getNx(), getNx(), M -> getLd(), getLd(), B -> getLd(), A_T, B_T);
    const dev_hierarchical <T> *h_a = getElementHierarchical();
    if (h_a != nullptr)
    { } //TODO
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