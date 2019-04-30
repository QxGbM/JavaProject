
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

  __host__ void setElement (void * element_in, element_t type_in)
  {
    element = element_in;
    type = type_in;
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
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GETRF(self); }
    if (lr != nullptr)
    { return lr -> generateOps_GETRF(self); }
    if (h != nullptr)
    { return h -> generateOps_GETRF(self); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_TRSML (const h_index *self, const dev_dense <T> *B, const h_index *index_b) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_TRSML(self, B, index_b); }
    if (lr != nullptr)
    { return lr -> generateOps_TRSML(self, B, index_b); }
    if (h != nullptr)
    { return h -> generateOps_TRSML(self, B, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_TRSML (const h_index *self, const dev_low_rank <T> *B, const h_index *index_b) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_TRSML(self, B, index_b); }
    if (lr != nullptr)
    { return lr -> generateOps_TRSML(self, B, index_b); }
    if (h != nullptr)
    { return h -> generateOps_TRSML(self, B, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_TRSML (const h_index *self, const dev_hierarchical <T> *B, const h_index *index_b) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_TRSML(self, B, index_b); }
    if (lr != nullptr)
    { return lr -> generateOps_TRSML(self, B, index_b); }
    if (h != nullptr)
    { return h -> generateOps_TRSML(self, B, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_TRSML (const h_index *self, const dev_h_element <T> *B, const h_index *index_b) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_TRSML(self, B, index_b); }
    if (lr != nullptr)
    { return lr -> generateOps_TRSML(self, B, index_b); }
    if (h != nullptr)
    { return h -> generateOps_TRSML(self, B, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_TRSMR (const h_index *self, const dev_dense <T> *B, const h_index *index_b) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_TRSMR(self, B, index_b); }
    if (lr != nullptr)
    { return lr -> generateOps_TRSMR(self, B, index_b); }
    if (h != nullptr)
    { return h -> generateOps_TRSMR(self, B, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_TRSMR (const h_index *self, const dev_low_rank <T> *B, const h_index *index_b) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_TRSMR(self, B, index_b); }
    if (lr != nullptr)
    { return lr -> generateOps_TRSMR(self, B, index_b); }
    if (h != nullptr)
    { return h -> generateOps_TRSMR(self, B, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_TRSMR (const h_index *self, const dev_hierarchical <T> *B, const h_index *index_b) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_TRSMR(self, B, index_b); }
    if (lr != nullptr)
    { return lr -> generateOps_TRSMR(self, B, index_b); }
    if (h != nullptr)
    { return h -> generateOps_TRSMR(self, B, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_TRSMR (const h_index *self, const dev_h_element <T> *B, const h_index *index_b) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_TRSMR(self, B, index_b); }
    if (lr != nullptr)
    { return lr -> generateOps_TRSMR(self, B, index_b); }
    if (h != nullptr)
    { return h -> generateOps_TRSMR(self, B, index_b); }

    return nullptr;
  }


  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_dense <T> *A, const h_index *index_a, const bool A_T, const dev_dense <T> *B, const h_index *index_b, const bool B_T) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_dense <T> *A, const h_index *index_a, const bool A_T, const dev_low_rank <T> *B, const h_index *index_b, const bool B_T) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }

    return nullptr;
  }
  
  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_dense <T> *A, const h_index *index_a, const bool A_T, const dev_hierarchical <T> *B, const h_index *index_b, const bool B_T) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_dense <T> *A, const h_index *index_a, const bool A_T, const dev_h_element <T> *B, const h_index *index_b, const bool B_T) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_low_rank <T> *A, const h_index *index_a, const bool A_T, const dev_dense <T> *B, const h_index *index_b, const bool B_T) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_low_rank <T> *A, const h_index *index_a, const bool A_T, const dev_low_rank <T> *B, const h_index *index_b, const bool B_T) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_low_rank <T> *A, const h_index *index_a, const bool A_T, const dev_hierarchical <T> *B, const h_index *index_b, const bool B_T) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_low_rank <T> *A, const h_index *index_a, const bool A_T, const dev_h_element <T> *B, const h_index *index_b, const bool B_T) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_hierarchical <T> *A, const h_index *index_a, const bool A_T, const dev_dense <T> *B, const h_index *index_b, const bool B_T) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_hierarchical <T> *A, const h_index *index_a, const bool A_T, const dev_low_rank <T> *B, const h_index *index_b, const bool B_T) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_hierarchical <T> *A, const h_index *index_a, const bool A_T, const dev_hierarchical <T> *B, const h_index *index_b, const bool B_T) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_hierarchical <T> *A, const h_index *index_a, const bool A_T, const dev_h_element <T> *B, const h_index *index_b, const bool B_T) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }

    return nullptr;
  }

    __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_h_element <T> *A, const h_index *index_a, const bool A_T, const dev_dense <T> *B, const h_index *index_b, const bool B_T) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_h_element <T> *A, const h_index *index_a, const bool A_T, const dev_low_rank <T> *B, const h_index *index_b, const bool B_T) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_h_element <T> *A, const h_index *index_a, const bool A_T, const dev_hierarchical <T> *B, const h_index *index_b, const bool B_T) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_h_element <T> *A, const h_index *index_a, const bool A_T, const dev_h_element <T> *B, const h_index *index_b, const bool B_T) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, A_T, B, index_b, B_T); }

    return nullptr;
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