
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
    dev_dense <T> *d = get_element_dense();
    dev_low_rank <T> *lr = get_element_low_rank();
    dev_hierarchical <T> *h = get_element_hierarchical();

    if (d != nullptr) { delete d; }
    if (lr != nullptr) { delete lr; }
    if (h != nullptr) { delete h; }
  }

  __host__ dev_dense <T> * get_element_dense() const
  {
    return (type == dense) ? ((dev_dense <T> *) element) : nullptr;
  }

  __host__ dev_low_rank <T> * get_element_low_rank() const
  {
    return (type == low_rank) ? ((dev_low_rank <T> *) element) : nullptr;
  }

  __host__ dev_hierarchical <T> * get_element_hierarchical() const
  {
    return (type == hierarchical) ? ((dev_hierarchical <T> *) element) : nullptr;
  }

  __host__ element_t getType() const
  {
    return type;
  }

  __host__ int getNx() const
  {
    const dev_dense <T> *d = get_element_dense();
    const dev_low_rank <T> *lr = get_element_low_rank();
    const dev_hierarchical <T> *h = get_element_hierarchical();

    if (d != nullptr)
    { return d -> getNx(); }
    if (lr != nullptr)
    {
      //TODO
    }
    if (h != nullptr)
    { return h -> getNx(); }

    return 0;
  }

  __host__ int getNy() const
  {
    const dev_dense <T> *d = get_element_dense();
    const dev_low_rank <T> *lr = get_element_low_rank();
    const dev_hierarchical <T> *h = get_element_hierarchical();

    if (d != nullptr)
    { return d -> getNy(); }
    if (lr != nullptr)
    {
      //TODO
    }
    if (h != nullptr)
    { return h -> getNy(); }

    return 0;
  }

  __host__ int getLd() const
  {
    const dev_dense <T> *d = get_element_dense();

    if (d != nullptr)
    { return d -> getLd(); }

    return 0;
  }

  __host__ int getRank() const
  {
    const dev_low_rank <T> *lr = get_element_low_rank();

    if (lr != nullptr)
    {
      //TODO
    }

    return 0;
  }

  __host__ dev_dense <T> * convertToDense() const
  {
    const dev_dense <T> *d = get_element_dense();
    const dev_low_rank <T> *lr = get_element_low_rank();
    const dev_hierarchical <T> *h = get_element_hierarchical();

    if (d != nullptr)
    {
      return new dev_dense <T> (d -> getNx(), d -> getNy(), d -> getElements(), d -> getLd());
    }
    if (lr != nullptr)
    {
      // TODO
    }
    if (h != nullptr)
    {
      return h -> convertToDense();
    }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GETRF (const h_index *self) const
  {
    h_ops_tree * ops = new h_ops_tree( new h_ops(getrf, self, getNx(), getNy(), getLd()) );
    const dev_hierarchical <T> *h = get_element_hierarchical();
    if (h != nullptr) { ops -> hookup_child (h -> generateOps_GETRF(self)); }
    return ops;
  }

  __host__ h_ops_tree * generateOps_TRSML (const h_index *self, const dev_h_element <T> *B, const h_index *index_b) const
  {
    h_ops_tree * ops = new h_ops_tree( new h_ops(trsml, self, index_b, getNx(), getNy(), B -> getNx(), getLd(), B -> getLd()) );
    const dev_hierarchical <T> *h = get_element_hierarchical();
    if (h != nullptr) {  }
    return ops;
  }

  __host__ h_ops_tree * generateOps_TRSMR (const h_index *self, const dev_h_element <T> *B, const h_index *index_b) const
  {
    h_ops_tree * ops = new h_ops_tree( new h_ops(trsmr, self, index_b, getNx(), getNy(), B -> getNy(), getLd(), B -> getLd()) );
    const dev_hierarchical <T> *h = get_element_hierarchical();
    if (h != nullptr) {  }
    return ops;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_h_element <T> *A, const h_index *index_a, const dev_h_element <T> *B, const h_index *index_b) const
  {
    h_ops_tree * ops = new h_ops_tree( new h_ops(gemm, self, index_a, index_b, getNx(), getNy(), A -> getNy(), getLd(), A -> getLd(), B->getLd()) );
    const dev_hierarchical <T> *h = get_element_hierarchical();
    if (h != nullptr) { }
    return ops;
  }

  __host__ void print (const h_index *index) const
  {
    const dev_dense <T> *d = get_element_dense();
    const dev_low_rank <T> *lr = get_element_low_rank();
    const dev_hierarchical <T> *h = get_element_hierarchical();

    index -> print();

    if (d != nullptr) { d -> print(); }
    if (lr != nullptr) { lr -> print(); }
    if (h != nullptr) { h -> print(index); } 
  }

};


#endif