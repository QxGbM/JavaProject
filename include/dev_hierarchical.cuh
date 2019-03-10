
#ifndef _DEV_HIERARCHICAL_CUH
#define _DEV_HIERARCHICAL_CUH

#include <dev_low_rank.cuh>
#include <index.cuh>

enum h_matrix_t {
  empty,
  dense,
  low_rank,
  hierarchical,
};

template <class matrixEntriesT> struct dev_hierarchical;

template <class matrixEntriesT> struct h_matrix_element {

  void *element;
  h_matrix_t element_type;
  
  __host__ h_matrix_element (void *element_in = nullptr, const h_matrix_t type_in = empty)
  {
    element = element_in;
    element_type = type_in;
  }

  __host__ struct dev_dense <matrixEntriesT> * get_element_dense () const
  { return (element_type == dense) ? ((struct dev_dense <matrixEntriesT> *) element) : nullptr; }

  __host__ struct dev_low_rank <matrixEntriesT> * get_element_low_rank() const
  { return (element_type == low_rank) ? ((struct dev_low_rank <matrixEntriesT> *) element) : nullptr; }

  __host__ struct dev_hierarchical <matrixEntriesT> * get_element_hierarchical () const
  { return (element_type == hierarchical) ? ((struct dev_hierarchical <matrixEntriesT> *) element) : nullptr; }

  __host__ ~h_matrix_element ()
  { 
    struct dev_dense <matrixEntriesT> *d = get_element_dense();
    struct dev_low_rank <matrixEntriesT> *lr = get_element_low_rank();
    struct dev_hierarchical <matrixEntriesT> *h = get_element_hierarchical();

    if (d != nullptr) { delete d; }
    else if (lr != nullptr) { delete lr; }
    else if (h != nullptr) { delete h; } 

  }

  __host__ int * getDim() const
  {
    int *dim = new int[2]{ 0, 0 };
    const struct dev_dense <matrixEntriesT> *d = get_element_dense();
    const struct dev_low_rank <matrixEntriesT> *lr = get_element_low_rank();
    const struct dev_hierarchical <matrixEntriesT> *h = get_element_hierarchical();

    if (d != nullptr) 
    {
      dim[0] = d -> nx;
      dim[1] = d -> ny;
    }
    else if (lr != nullptr) 
    {
      // TODO
    }
    else if (h != nullptr) 
    {
      int *dim_h = h -> getDim();
      dim[0] = dim_h[0];
      dim[1] = dim_h[1];
    }
    return dim;
  }

  __host__ void print() const
  {

    const struct dev_dense <matrixEntriesT> *d = get_element_dense();
    const struct dev_low_rank <matrixEntriesT> *lr = get_element_low_rank();
    const struct dev_hierarchical <matrixEntriesT> *h = get_element_hierarchical();

    if (d != nullptr) { d -> print(); }
    else if (lr != nullptr) { lr -> print(); }
    else if (h != nullptr) { h -> print(); } 
  }

};

template <class matrixEntriesT> struct dev_hierarchical {

  int nx;
  int ny;
  struct h_matrix_element <matrixEntriesT> **elements;
  
  __host__ dev_hierarchical (const int x, const int y)
  {
    nx = x;
    ny = y;
    elements = new struct h_matrix_element <matrixEntriesT> * [x * y];
    for (int i = 0; i < x * y; i++) { elements[i] = nullptr; }
  }

  __host__ ~dev_hierarchical ()
  {
    for (int i = 0; i < nx * ny; i++)
    { 
      if (elements[i] != nullptr) 
      { delete elements[i]; } 
    }
    delete[] elements;
  }

  __host__ void set_element (void *matrix, const h_matrix_t type, const int x, const int y) 
  {
    elements[y * nx + x] = new h_matrix_element <matrixEntriesT> (matrix, type);
  }

  __host__ void print () const
  {
    for (int i = 0; i < ny * nx; i++)
    {
      if (elements[i] != nullptr) 
      { elements[i] -> print(); }
    }
  }

  __host__ int * getDim() const
  {
    int *dim = new int[2]{ 0, 0 };
    for (int i = 0; i < ny * nx; i++)
    {
      if (elements[i] != nullptr)
      { 
        int *dim_e = elements[i] -> getDim();
        dim[0] += dim_e[0];
        dim[1] += dim_e[1];
        delete[] dim_e;
      }
    }
    return dim;
  }

  __host__ struct h_matrix_element <matrixEntriesT> * lookup (const int *n, const int levels) const
  {
    if (levels == 1)
    { return elements[n[0]]; }
    else
    {
      struct dev_hierarchical *h = elements[n[0]] -> get_element_hierarchical();
      return (h == nullptr) ? nullptr : h -> lookup(&n[1], levels - 1);
    }
  }

  __host__ struct h_matrix_element <matrixEntriesT> * lookup (const struct multi_level_index *i) const
  {
    return lookup (i -> ns, i -> levels);
  }

  __host__ matrixEntriesT * lookup_element (const int *n, const int levels, const int offset) const
  {
    const struct h_matrix_element <matrixEntriesT> e = lookup(n, levels);
    const struct dev_dense <matrixEntriesT> d = e -> get_element_dense();
    const struct dev_low_rank <matrixEntriesT> lr = e -> get_element_low_rank();
    const struct dev_hierarchical <matrixEntriesT> h = e -> get_element_hierarchical();
    if (d != nullptr)
    { return &(d -> elements)[offset]; }
    else if (lr != nullptr)
    { return nullptr; } // TODO
    else
    { return nullptr; }
  }

  __host__ matrixEntriesT * lookup_element_dev (const struct multi_level_index *i) const
  {
    return lookup_element_dev (i -> ns, i -> levels, i -> offset);
  }

  __host__ matrixEntriesT * lookup_element_dev (const int *n, const int levels, const int offset) const
  {
    const struct h_matrix_element <matrixEntriesT> e = lookup(n, levels);
    const struct dev_dense <matrixEntriesT> d = e -> get_element_dense();
    const struct dev_low_rank <matrixEntriesT> lr = e -> get_element_low_rank();
    const struct dev_hierarchical <matrixEntriesT> h = e -> get_element_hierarchical();
    if (d != nullptr)
    { return &(d -> dev_ptr)[offset]; }
    else if (lr != nullptr)
    { return nullptr; } // TODO
    else
    { return nullptr; }
  }

  __host__ matrixEntriesT * lookup_element (const struct multi_level_index *i) const
  {
    return lookup_element (i -> ns, i -> levels, i -> offset);
  }

  __host__ void loadTestMatrix (const int levels = 1, const int dim = 2, const int block_size = 4)
  {
    for (int y = 0; y < ny; y++)
    {
      for (int x = 0; x < nx; x++)
      {
        if (x == y && levels > 0)
        { 
          struct dev_hierarchical <matrixEntriesT> *e = new dev_hierarchical <matrixEntriesT> (dim, dim);
          e -> loadTestMatrix(levels - 1, dim, block_size); 
          set_element(e, hierarchical, x, y);
        }
        else
        {
          int l = block_size, cl = levels; 
          while (cl > 0) { l *= dim; cl--; }
          struct dev_dense <matrixEntriesT> *e = new dev_dense <matrixEntriesT> (l, l);
          e -> loadRandomMatrix(-10, 10);
          set_element(e, dense, x, y);
        }
      }
    }

  }

};

#endif