
#ifndef _DEV_HIERARCHICAL_CUH
#define _DEV_HIERARCHICAL_CUH

#include <dev_dense.cuh>
#include <index.cuh>

enum h_matrix_t {
  empty,
  dense,
  low_rank,
  hierarchical,
};

template <class matrixEntriesT> struct dev_hierarchical;

template <class matrixEntriesT> struct h_matrix_element {

  struct multi_level_index *index;
  h_matrix_t element_type;
  void *element;

  __host__ h_matrix_element (const int n = 0, const struct multi_level_index *parent = nullptr, const h_matrix_t type = empty, void *e = nullptr)
  {
    index = new multi_level_index(n, parent);
    element_type = type;
    element = e;
  }

  __host__ ~h_matrix_element ()
  {
    index -> ~multi_level_index();
    free(index);

    switch (element_type)
    {
      case empty:
      { break; }
      case dense: 
      {
        ((struct dev_dense <matrixEntriesT> *) element) -> ~dev_dense(); 
        break;
      }
      case low_rank: 
      {
        //TODO 
        break;
      }
      case hierarchical:
      { 
        ((struct dev_hierarchical <matrixEntriesT> *) element) -> ~dev_hierarchical(); 
        break;
      }
    }
    free(element);
  }

  __host__ void print()
  {
    index -> print();
    switch (element_type)
    {
      case empty:
      {
        printf("empty\n\n");
        break;
      }
      case dense: 
      {
        ((struct dev_dense <matrixEntriesT> *) element) -> print(); 
        break;
      }
      case low_rank: 
      {
        //TODO 
        break;
      }
      case hierarchical: 
      {
        ((struct dev_hierarchical <matrixEntriesT> *) element) -> print(); 
        break;
      }
    }
  }

};

template <class matrixEntriesT> struct dev_hierarchical {

  int nx;
  int ny;
  struct h_matrix_element <matrixEntriesT> **elements;
  struct multi_level_index *index;
  
  __host__ dev_hierarchical (const int x, const int y, struct multi_level_index *i = nullptr)
  {
    nx = x;
    ny = y;
    elements = (struct h_matrix_element <matrixEntriesT> **) malloc (x * y * sizeof(struct h_matrix_element <matrixEntriesT> *));
    memset (elements, 0, x * y * sizeof(struct h_matrix_element <matrixEntriesT> *));
    index = i;
  }

  __host__ ~dev_hierarchical ()
  {
    for (int i = 0; i < nx * ny; i++)
    { 
      if (elements[i] != nullptr) 
      { elements[i] -> ~h_matrix_element(); free(elements[i]); } 
    }
    free(elements);

    if (index != nullptr)
    {
      index -> ~multi_level_index();
      free(index);
    }
  }

  __host__ void set_element(void *matrix, h_matrix_t type, int x, int y) 
  {
    elements[y * nx + x] = new h_matrix_element <matrixEntriesT> (y * nx + x, index, type, (void *) matrix);
  }

  __host__ void print()
  {
    for (int y = 0; y < ny; y++)
    {
      for (int x = 0; x < nx; x++)
      {
        if (elements[y * nx + x] != nullptr) 
        {
          elements[y * nx + x] -> print();
        }
      }
    }
  }

  __host__ void loadTestMatrix (const int levels = 1, const int dim = 2, const int block_size = 4)
  {
    for (int y = 0; y < ny; y++)
    {
      for (int x = 0; x < nx; x++)
      {
        if (x == y && levels > 0)
        { 
          struct multi_level_index *i = new multi_level_index(y * nx + x, index);
          struct dev_hierarchical <matrixEntriesT> *e = new dev_hierarchical <matrixEntriesT> (dim, dim, i);
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

  __host__ struct h_matrix_element <matrixEntriesT> * lookup (const int *n, const int levels = 1)
  {
    if (levels == 1)
    { return elements[n[0]]; }
    else
    {
      if (elements[n[0]] -> element_type == hierarchical)
      {
        struct dev_hierarchical <matrixEntriesT> *p = (struct dev_hierarchical <matrixEntriesT> *) (elements[n[0]] -> element);
        return p -> lookup(&n[1], levels - 1);
      }
      else
      {
        return nullptr;
      }
    }
  }

  __host__ struct h_matrix_element <matrixEntriesT> * lookup (const struct multi_level_index *i)
  {
    return lookup (i -> ns, i -> levels);
  }

};

#endif