
#ifndef _DEV_HIERARCHICAL_CUH
#define _DEV_HIERARCHICAL_CUH

#include <dev_low_rank.cuh>
#include <index.cuh>

template <class T> class h_matrix_element 
{
private:

  void *element;
  h_matrix_t element_type;

public:
  
  __host__ h_matrix_element (void *element_in = nullptr, const h_matrix_t type_in = empty)
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

  __host__ h_matrix_t getType() const
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

template <class T> class dev_hierarchical 
{
private:

  int nx;
  int ny;
  h_matrix_element <T> **elements;

public:
  
  __host__ dev_hierarchical (const int x, const int y)
  {
    nx = x;
    ny = y;
    elements = new h_matrix_element <T> * [x * y];
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
    elements[y * nx + x] = new h_matrix_element <T> (matrix, type);
  }

  __host__ int * getDim3(const bool actual = true) const
  {
    int *dim = new int[3]{ 0, 0, 0 };
    if (actual)
    {
      for (int i = 0; i < ny * nx; i++)
      {
        if (elements[i] != nullptr)
        {
          int *dim_e = elements[i]->getDim3();
          dim[0] += dim_e[0];
          dim[1] += dim_e[1];
          delete[] dim_e;
        }
      }
      dim[0] /= ny;
      dim[1] /= nx;
    }
    else
    {
      dim[0] = nx;
      dim[1] = ny;
    }
    return dim;
  }

  __host__ void print (const multi_level_index *index_in = nullptr) const
  {
    const int *dim = getDim3(false);
    for (int i = 0; i < ny * nx; i++)
    {
      if (elements[i] != nullptr) 
      {
        const multi_level_index *index = (index_in == nullptr) ? new multi_level_index(0, nullptr, i, 0, dim) : index_in -> child(i, 0, dim);
        elements[i] -> print(index);
        delete index;
      }
    }
    delete[] dim;
  }

  __host__ h_matrix_element <T> * lookup (const int i) const
  {
    return elements[i];
  }

  __host__ h_matrix_element <T> * lookup (const int levels, const int *n) const
  {
    const h_matrix_element <T> *e = lookup(n[0]);
    if (levels == 1)
    { return e; }
    else
    {
      const dev_hierarchical <T> *h = e -> get_element_hierarchical();
      return (h == nullptr) ? nullptr : h -> lookup(levels - 1, &n[1]);
    }
  }

  __host__ h_matrix_element <T> * lookup (const multi_level_index *i) const
  {
    return lookup (i -> levels, i -> ns);
  }

  __host__ multi_level_index * child_index (const int child_id, const multi_level_index *self_index = nullptr) const
  {
    const h_matrix_element <T> *child = lookup(child_id);
    const int *dim = child -> getDim3(true);
    multi_level_index *index = (self_index == nullptr) ? new multi_level_index(0, nullptr, child_id, 0, dim) : self_index -> child(child_id, 0, dim);
    delete[] dim;
    return index;
  }

  __host__ void loadTestMatrix (const int levels = 1, const int dim = 2, const int block_size = 4)
  {
    for (int y = 0; y < ny; y++)
    {
      for (int x = 0; x < nx; x++)
      {
        if (x == y && levels > 0)
        { 
          dev_hierarchical <T> *e = new dev_hierarchical <T> (dim, dim);
          e -> loadTestMatrix(levels - 1, dim, block_size); 
          set_element(e, hierarchical, x, y);
        }
        else
        {
          int l = block_size, cl = levels; 
          while (cl > 0) { l *= dim; cl--; }
          dev_dense <T> *e = new dev_dense <T> (l, l);
          e -> loadRandomMatrix(-10, 10);
          set_element(e, dense, x, y);
        }
      }
    }

  }

};

#endif