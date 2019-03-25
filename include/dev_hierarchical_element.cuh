
#ifndef _DEV_HIERARCHICAL_ELEMENT_CUH
#define _DEV_HIERARCHICAL_ELEMENT_CUH

#include <pspl.cuh>

template <class T> class h_matrix_element 
{
private:

  void *element;
  element_t element_type;

public:
  
  __host__ h_matrix_element (void *element_in = nullptr, const element_t type_in = empty)
  {
    element = element_in;
    element_type = type_in;
  }

  __host__ dev_dense <T> * get_element_dense () const
  { return (element_type == dense) ? ((dev_dense <T> *) element) : nullptr; }

  __host__ dev_low_rank <T> * get_element_low_rank() const
  { return (element_type == low_rank) ? ((dev_low_rank <T> *) element) : nullptr; }

  __host__ dev_hierarchical <T> * get_element_hierarchical () const
  { return (element_type == hierarchical) ? ((dev_hierarchical <T> *) element) : nullptr; }

  __host__ ~h_matrix_element ()
  { 
    dev_dense <T> *d = get_element_dense();
    dev_low_rank <T> *lr = get_element_low_rank();
    dev_hierarchical <T> *h = get_element_hierarchical();

    if (d != nullptr) { delete d; }
    else if (lr != nullptr) { delete lr; }
    else if (h != nullptr) { delete h; }
  }

  __host__ element_t getType() const
  {
    return element_type;
  }

  __host__ int * getDim3 (const bool actual = true) const
  {
    const dev_dense <T> *d = get_element_dense();
    const dev_low_rank <T> *lr = get_element_low_rank();
    const dev_hierarchical <T> *h = get_element_hierarchical();

    if (d != nullptr) 
    { return d -> getDim3(); }
    else if (lr != nullptr) 
    {
      // TODO
      return new int[3]{ 0, 0, 0 };
    }
    else if (h != nullptr) 
    {
      int *dim = new int[3]{ 0, 0, 0 };
      int *dim_h = h -> getDim3(actual);
      dim[0] = dim_h[0];
      dim[1] = dim_h[1];
      dim[2] = dim_h[2];
      delete[] dim_h;
      return dim;
    }
    else
    { return new int[3]{ 0, 0, 0 }; }
  }

  __host__ void print(const multi_level_index *index) const
  {
    const dev_dense <T> *d = get_element_dense();
    const dev_low_rank <T> *lr = get_element_low_rank();
    const dev_hierarchical <T> *h = get_element_hierarchical();

    index -> print();

    if (d != nullptr) { d -> print(); }
    else if (lr != nullptr) { lr -> print(); }
    else if (h != nullptr) { h -> print(index); } 
  }

};


#endif