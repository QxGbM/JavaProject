
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

  __host__ inline dev_dense <T> * getElementDense() const
  {
    return (type == dense) ? ((dev_dense <T> *) element) : nullptr;
  }

  __host__ inline dev_low_rank <T> * getElementLowRank() const
  {
    return (type == low_rank) ? ((dev_low_rank <T> *) element) : nullptr;
  }

  __host__ inline dev_hierarchical <T> * getElementHierarchical() const
  {
    return (type == hierarchical) ? ((dev_hierarchical <T> *) element) : nullptr;
  }

  __host__ inline element_t getType() const
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
    { return h -> getNx_abs(); }

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
    { return h -> getNy_abs(); }

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

  __host__ T getElement (const int y_in, const int x_in) const
  {
    const dev_dense <T> *d = getElementDense();
    const dev_low_rank <T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return (d -> getElements())[y_in * (d -> getLd()) + x_in]; }
    if (lr != nullptr)
    { return lr -> getElement(y_in, x_in); }
    if (h != nullptr)
    { return h -> getElement_abs(y_in, x_in); }

    return 0;
  }

  __host__ bool partitionAccording (dev_h_element <T> * y = nullptr, dev_h_element <T> * x = nullptr)
  {
    dev_hierarchical <T> *h_y = (y == nullptr) ? nullptr : y -> getElementHierarchical();
    dev_hierarchical <T> *h_x = (x == nullptr) ? nullptr : x -> getElementHierarchical();
    if (h_y == nullptr && h_x == nullptr)
    { return true; }

    dev_dense <T> *d = getElementDense();
    if (d != nullptr)
    { return true; }

    const int ny_i = (h_y == nullptr) ? 1 : h_y -> getNy_blocks(), nx_i = (h_x == nullptr) ? 1 : h_x -> getNx_blocks();
    if (ny_i == 1 && nx_i == 1)
    { return true; }

    int * y_offsets = nullptr, * x_offsets = nullptr;
    if (h_y != nullptr) { h_y -> getOffsets_y(&y_offsets); }
    if (h_x != nullptr) { h_x -> getOffsets_x(&x_offsets); }

    dev_low_rank <T> *lr = getElementLowRank();
    dev_hierarchical <T> *h;

    if (lr != nullptr)
    {
      dev_low_rank <T> ** list = lr -> createPartitions (ny_i, y_offsets, nx_i, x_offsets);
      h = new dev_hierarchical <T> (nx_i, ny_i, low_rank, list);
      delete lr;
      element = h; type = hierarchical;
    }
    else
    { h = getElementHierarchical(); }

    if (h != nullptr)
    {
      const int ny_h = h -> getNy_blocks(), nx_h = h -> getNx_blocks();
      int * y_offsets_h, * x_offsets_h;
      bool success = true;
      h -> getOffsets_y(&y_offsets_h);
      h -> getOffsets_x(&x_offsets_h);

      if ((ny_h != ny_i) || (nx_h != nx_i))
      { 
        printf("-- Partition Failed: Hierarchical Matrices are already partioned in a different way. y: %d vs %d, x: %d vs %d. --\n", ny_h, ny_i, nx_h, nx_i); 
        success = false;
      }
      else
      {
        for (int i = 0; i < ny_i + 1 && success; i++)
        {
          if (y_offsets[i] != y_offsets_h[i]) 
          { success = false; }
        }

        for (int i = 0; i < nx_i + 1 && success; i++)
        {
          if (x_offsets[i] != x_offsets_h[i]) 
          { success = false; }
        }

        if (!success)
        { printf("-- Partition Failed: Hierarchical Matrices are already partioned in a different way. y: %d vs %d, x: %d vs %d. --\n"); }
      }

      for (int i = 0; i < ny_i && success; i++) for (int j = 0; j < nx_i && success; j++)
      { success = h -> getElement_blocks(y, x) -> partitionAccording(h_y -> getElement_blocks(y, 0), h_x -> getElement_blocks(0, x)); }

      delete[] y_offsets_h;
      delete[] x_offsets_h;
      delete[] y_offsets;
      delete[] x_offsets;
      return success;
    }
    else
    {
      delete[] y_offsets;
      delete[] x_offsets;
      return true;
    }

  }

  __host__ dev_dense <T> * convertToDense() const
  {
    const dev_dense <T> *d = getElementDense();
    const dev_low_rank <T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d; }
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


  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_dense <T> *A, const h_index *index_a, const dev_dense <T> *B, const h_index *index_b) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, B, index_b); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, B, index_b); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, B, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_dense <T> *A, const h_index *index_a, const dev_low_rank <T> *B, const h_index *index_b) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, B, index_b); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, B, index_b); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, B, index_b); }

    return nullptr;
  }
  
  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_dense <T> *A, const h_index *index_a, const dev_hierarchical <T> *B, const h_index *index_b) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, B, index_b); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, B, index_b); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, B, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_dense <T> *A, const h_index *index_a, const dev_h_element <T> *B, const h_index *index_b) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, B, index_b); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, B, index_b); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, B, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_low_rank <T> *A, const h_index *index_a, const dev_dense <T> *B, const h_index *index_b) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, B, index_b); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, B, index_b); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, B, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_low_rank <T> *A, const h_index *index_a, const dev_low_rank <T> *B, const h_index *index_b) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, B, index_b); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, B, index_b); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, B, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_low_rank <T> *A, const h_index *index_a, const dev_hierarchical <T> *B, const h_index *index_b) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, B, index_b); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, B, index_b); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, B, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_low_rank <T> *A, const h_index *index_a, const dev_h_element <T> *B, const h_index *index_b) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, B, index_b); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, B, index_b); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, B, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_hierarchical <T> *A, const h_index *index_a, const dev_dense <T> *B, const h_index *index_b) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, B, index_b); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, B, index_b); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, B, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_hierarchical <T> *A, const h_index *index_a, const dev_low_rank <T> *B, const h_index *index_b) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, B, index_b); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, B, index_b); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, B, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_hierarchical <T> *A, const h_index *index_a, const dev_hierarchical <T> *B, const h_index *index_b) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, B, index_b); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, B, index_b); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, B, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_hierarchical <T> *A, const h_index *index_a, const dev_h_element <T> *B, const h_index *index_b) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, B, index_b); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, B, index_b); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, B, index_b); }

    return nullptr;
  }

    __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_h_element <T> *A, const h_index *index_a, const dev_dense <T> *B, const h_index *index_b) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, B, index_b); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, B, index_b); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, B, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_h_element <T> *A, const h_index *index_a, const dev_low_rank <T> *B, const h_index *index_b) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, B, index_b); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, B, index_b); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, B, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_h_element <T> *A, const h_index *index_a, const dev_hierarchical <T> *B, const h_index *index_b) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, B, index_b); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, B, index_b); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, B, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_h_element <T> *A, const h_index *index_a, const dev_h_element <T> *B, const h_index *index_b) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, B, index_b); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, B, index_b); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, B, index_b); }

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