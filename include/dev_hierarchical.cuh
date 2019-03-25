
#ifndef _DEV_HIERARCHICAL_CUH
#define _DEV_HIERARCHICAL_CUH

#include <pspl.cuh>

template <class T> class dev_hierarchical 
{
private:

  int nx;
  int ny;
  dev_h_element <T> * elements;

public:
  
  __host__ dev_hierarchical (const int nx_in, const int ny_in)
  {
    nx = nx_in;
    ny = ny_in;
    elements = new dev_h_element <T> [nx * ny];
    for (int y = 0; y < ny; y++)
    {
      for (int x = 0; x < nx; x++)
      { setElement(nullptr, empty, x, y); }
    }
  }

  __host__ ~dev_hierarchical ()
  {
    delete[] elements;
  }

  __host__ void setElement (void * M, const element_t type, const int x, const int y) 
  {
    if (x < nx && y < ny)
    { elements[y * nx + x] = * new dev_h_element <T>(M, type); }
  }

  __host__ h_index * getRootIndex () const
  {
    return new h_index (0, nullptr, 0, this);
  }

  __host__ bool dimIntegrityCheck () const
  { 
    for (int y = 0; y < ny; y++)
    {
      const int rows = elements[y * nx].getNy();
      for (int x = 1; x < nx; x++)
      {
        const int rows_x = elements[y * nx + x].getNy();
        if (rows != rows_x) { return false; }
      }
    }

    for (int x = 0; x < nx; x++)
    {
      const int cols = elements[x].getNx();
      for (int y = 1; y < ny; y++)
      {
        const int cols_y = elements[y * nx + x].getNx();
        if (cols != cols_y) { return false; }
      }
    }

    return true;
  }

  __host__ int getNx () const
  {
    if (dimIntegrityCheck())
    {
      int n = 0;
      for (int i = 0; i < nx; i++)
      {
        n += elements[i].getNx();
      }
      return n;
    }
    return 0;
  }

  __host__ int getNy () const
  {
    if (dimIntegrityCheck())
    {
      int n = 0;
      for (int i = 0; i < ny; i++)
      {
        n += elements[i * nx].getNy();
      }
      return n;
    }
    return 0;
  }

  __host__ void print (const h_index * index_in) const
  {
    for (int i = 0; i < ny * nx; i++)
    {
      const h_index * i_index = index_in -> child(i);
      elements[i].print(i_index);
      delete i_index;
    }
  }

  __host__ void print() const
  {
    const h_index * root = getRootIndex();
    print(root);
    delete root;
  }

  __host__ dev_h_element <T> * lookup (const int i) const
  {
    return &elements[i];
  }

  __host__ dev_h_element <T> * lookup (const int levels, const int *n) const
  {
    const dev_h_element <T> *e = lookup (n[0]);
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

  __host__ void loadTestMatrix (const int levels, const int dim, const int block_size)
  {
    for (int y = 0; y < ny; y++)
    {
      for (int x = 0; x < nx; x++)
      {
        if (x == y && levels > 0)
        { 
          dev_hierarchical <T> *e = new dev_hierarchical <T> (dim, dim);
          e -> loadTestMatrix(levels - 1, dim, block_size); 
          setElement(e, hierarchical, x, y);
        }
        else
        {
          int l = block_size, cl = levels; 
          while (cl > 0) { l *= dim; cl--; }
          dev_dense <T> *e = new dev_dense <T> (l, l);
          e -> loadRandomMatrix(-10, 10);
          setElement(e, dense, x, y);
        }
      }
    }

  }

};

#endif