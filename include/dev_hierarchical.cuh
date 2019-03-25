
#ifndef _DEV_HIERARCHICAL_CUH
#define _DEV_HIERARCHICAL_CUH

#include <pspl.cuh>

template <class T> class dev_hierarchical 
{
private:

  int nx;
  int ny;
  dev_h_element <T> **elements;

public:
  
  __host__ dev_hierarchical (const int x, const int y)
  {
    nx = x;
    ny = y;
    elements = new dev_h_element <T> * [x * y];
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

  __host__ void set_element (void *matrix, const element_t type, const int x, const int y) 
  {
    elements[y * nx + x] = new dev_h_element <T> (matrix, type);
  }

  __host__ int * getDim3 (const bool actual = true) const
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

  __host__ void print (const h_index *index_in = nullptr) const
  {
    const int *dim = getDim3(false);
    for (int i = 0; i < ny * nx; i++)
    {
      if (elements[i] != nullptr) 
      {
        const h_index *index = (index_in == nullptr) ? new h_index(0, nullptr, i, 0, dim) : index_in -> child(i, 0, dim);
        elements[i] -> print(index);
        delete index;
      }
    }
    delete[] dim;
  }

  __host__ dev_h_element <T> * lookup (const int i) const
  {
    return elements[i];
  }

  __host__ dev_h_element <T> * lookup (const int levels, const int *n) const
  {
    const dev_h_element <T> *e = lookup(n[0]);
    if (levels == 1)
    { return e; }
    else
    {
      const dev_hierarchical <T> *h = e -> get_element_hierarchical();
      return (h == nullptr) ? nullptr : h -> lookup(levels - 1, &n[1]);
    }
  }

  __host__ dev_h_element <T> * lookup (const h_index *i) const
  {
    return lookup (i -> levels, i -> ns);
  }

  __host__ h_index * child_index (const int child_id, const h_index *self_index = nullptr) const
  {
    const dev_h_element <T> *child = lookup(child_id);
    const int *dim = child -> getDim3(true);
    h_index *index = (self_index == nullptr) ? new h_index(0, nullptr, child_id, 0, dim) : self_index -> child(child_id, 0, dim);
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