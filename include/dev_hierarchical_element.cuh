
#pragma once
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

  __host__ h_ops_tree * generateOps_GETRF (const h_index * self, dev_temp * tmp_mngr) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GETRF(self, tmp_mngr); }
    if (lr != nullptr)
    { return lr -> generateOps_GETRF(self, tmp_mngr); }
    if (h != nullptr)
    { return h -> generateOps_GETRF(self, tmp_mngr); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_TRSML (const h_index * self, const dev_dense <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_TRSML(self, B, index_b, tmp_mngr); }
    if (lr != nullptr)
    { return lr -> generateOps_TRSML(self, B, index_b, tmp_mngr); }
    if (h != nullptr)
    { return h -> generateOps_TRSML(self, B, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_TRSML (const h_index * self, const dev_low_rank <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_TRSML(self, B, index_b, tmp_mngr); }
    if (lr != nullptr)
    { return lr -> generateOps_TRSML(self, B, index_b, tmp_mngr); }
    if (h != nullptr)
    { return h -> generateOps_TRSML(self, B, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_TRSML (const h_index * self, const dev_hierarchical <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_TRSML(self, B, index_b, tmp_mngr); }
    if (lr != nullptr)
    { return lr -> generateOps_TRSML(self, B, index_b, tmp_mngr); }
    if (h != nullptr)
    { return h -> generateOps_TRSML(self, B, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_TRSML (const h_index * self, const dev_h_element <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_TRSML(self, B, index_b, tmp_mngr); }
    if (lr != nullptr)
    { return lr -> generateOps_TRSML(self, B, index_b, tmp_mngr); }
    if (h != nullptr)
    { return h -> generateOps_TRSML(self, B, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_TRSMR (const h_index * self, const dev_dense <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_TRSMR(self, B, index_b, tmp_mngr); }
    if (lr != nullptr)
    { return lr -> generateOps_TRSMR(self, B, index_b, tmp_mngr); }
    if (h != nullptr)
    { return h -> generateOps_TRSMR(self, B, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_TRSMR (const h_index * self, const dev_low_rank <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_TRSMR(self, B, index_b, tmp_mngr); }
    if (lr != nullptr)
    { return lr -> generateOps_TRSMR(self, B, index_b, tmp_mngr); }
    if (h != nullptr)
    { return h -> generateOps_TRSMR(self, B, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_TRSMR (const h_index * self, const dev_hierarchical <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_TRSMR(self, B, index_b, tmp_mngr); }
    if (lr != nullptr)
    { return lr -> generateOps_TRSMR(self, B, index_b, tmp_mngr); }
    if (h != nullptr)
    { return h -> generateOps_TRSMR(self, B, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_TRSMR (const h_index * self, const dev_h_element <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_TRSMR(self, B, index_b, tmp_mngr); }
    if (lr != nullptr)
    { return lr -> generateOps_TRSMR(self, B, index_b, tmp_mngr); }
    if (h != nullptr)
    { return h -> generateOps_TRSMR(self, B, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_ACCM (const h_index * self, const h_index * index_tmp_lr) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_ACCM(self, index_tmp_lr); }
    if (lr != nullptr)
    { return lr -> generateOps_ACCM(self, index_tmp_lr); }
    if (h != nullptr)
    { return h -> generateOps_ACCM(self, index_tmp_lr); }

    return nullptr;
  }


  __host__ h_ops_tree * generateOps_GEMM (const h_index * self, const dev_dense <T> * A, const h_index * index_a, const dev_dense <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index * self, const dev_dense <T> * A, const h_index * index_a, const dev_low_rank <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }

    return nullptr;
  }
  
  __host__ h_ops_tree * generateOps_GEMM (const h_index * self, const dev_dense <T> * A, const h_index * index_a, const dev_hierarchical <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index * self, const dev_dense <T> * A, const h_index * index_a, const dev_h_element <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index * self, const dev_low_rank <T> * A, const h_index * index_a, const dev_dense <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index * self, const dev_low_rank <T> * A, const h_index * index_a, const dev_low_rank <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index * self, const dev_low_rank <T> * A, const h_index * index_a, const dev_hierarchical <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index * self, const dev_low_rank <T> * A, const h_index * index_a, const dev_h_element <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index * self, const dev_hierarchical <T> * A, const h_index * index_a, const dev_dense <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index * self, const dev_hierarchical <T> * A, const h_index * index_a, const dev_low_rank <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index * self, const dev_hierarchical <T> * A, const h_index * index_a, const dev_hierarchical <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index * self, const dev_hierarchical <T> * A, const h_index * index_a, const dev_h_element <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }

    return nullptr;
  }

    __host__ h_ops_tree * generateOps_GEMM (const h_index * self, const dev_h_element <T> * A, const h_index * index_a, const dev_dense <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index * self, const dev_h_element <T> * A, const h_index * index_a, const dev_low_rank <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index * self, const dev_h_element <T> * A, const h_index * index_a, const dev_hierarchical <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index * self, const dev_h_element <T> * A, const h_index * index_a, const dev_h_element <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    const dev_dense<T> *d = getElementDense();
    const dev_low_rank<T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
    if (lr != nullptr)
    { return lr -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
    if (h != nullptr)
    { return h -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ cudaError_t loadBinary (FILE * stream, const bool reverse_bytes = true)
  {
    dev_dense<T> *d = getElementDense();
    dev_low_rank<T> *lr = getElementLowRank();
    dev_hierarchical <T> *h = getElementHierarchical();

    if (d != nullptr)
    { return d -> loadBinary(stream, reverse_bytes); }
    if (lr != nullptr)
    { return lr -> loadBinary(stream, reverse_bytes); }
    if (h != nullptr)
    { return h -> loadBinary(stream, reverse_bytes); }

    return cudaErrorMissingConfiguration;
  }

  __host__ static void * readStructureFromFile (FILE * stream, element_t * type, const int shadow_rank = _SHADOW_RANK)
  {
    char * buf = new char[32];
    if (stream != nullptr && fgets(buf, 32, stream) > 0)
    {
      int ny, nx, rank;

      if (buf[0] == 'H')
      { 
        * type = hierarchical;
        sscanf(buf, "H %d %d\n", &ny, &nx);
        delete[] buf;

        dev_hierarchical<T> * h = new dev_hierarchical<T> (nx, ny);

        for (int i = 0; i < ny; i++) for (int j = 0; j < nx; j++)
        {
          element_t type;
          void * element = readStructureFromFile (stream, &type, shadow_rank);
          h -> setElement(element, type, j, i);
        }

        h -> updateOffsets();
        return h;
      }
      else if (buf[0] == 'D')
      { 
        * type = dense; 
        sscanf(buf, "D %d %d\n", &ny, &nx);
        delete[] buf;
        dev_dense <T> * d = new dev_dense <T> (nx, ny);
        d -> resizeShadow (shadow_rank);
        return d;
      }
      else if (buf[0] == 'L' && buf[1] == 'R')
      { 
        * type = low_rank; 
        sscanf(buf, "LR %d %d %d\n", &ny, &nx, &rank);
        delete[] buf;
        return new dev_low_rank <T> (nx, ny, rank);
      }
      else
      { 
        * type = empty; 
        ny = nx = rank = 0;
        delete[] buf;
        return nullptr;
      }
    }
    else
    {
      printf("Error Reading from File.\n");
      delete[] buf;
      return nullptr;
    }

  }


  __host__ void print (const h_index * index) const
  {

    const dev_dense <T> *d = getElementDense();
    const dev_low_rank <T> *lr = getElementLowRank();
    const dev_hierarchical <T> *h = getElementHierarchical();

    index -> print();

    if (d != nullptr) { d -> print(0, 8, 0, 8); }
    if (lr != nullptr) { lr -> print(0, 8, 0, 8, 8); }
    if (h != nullptr) { h -> print(index); } 
  }

};


#endif