
#ifndef _DEV_HIERARCHICAL_CUH
#define _DEV_HIERARCHICAL_CUH

#include <dev_dense.cuh>

enum h_matrix_t {
  empty,
  dense,
  low_rank,
  hierarchical,
};

template <class matrixEntriesT> struct dev_hierarchical {

  int nx;
  int ny;
  void **elements;
  h_matrix_t *elements_type;
  
  __host__ dev_hierarchical (const int x, const int y)
  {
    nx = x;
    ny = y;
    elements = (void **) malloc (x * y * sizeof(void *));
    elements_type = (h_matrix_t *) malloc (x * y * sizeof(h_matrix_t *));

    memset (elements, 0, x * y * sizeof(void *));
    memset (elements_type, 0, x * y * sizeof(h_matrix_t *));
  }

  __host__ ~dev_hierarchical ()
  {
    for (int i = 0; i < nx * ny; i++)
    { 
      if (elements[i] != nullptr) 
      { 
        switch (elements_type[i])
        {
          case empty:
            break;
          case dense: 
            ((struct dev_dense <matrixEntriesT> *) elements[i]) -> ~dev_dense(); 
            break;
          case low_rank: 
            //TODO 
            break;
          case hierarchical: 
            ((struct dev_hierarchical <matrixEntriesT> *) elements[i]) -> ~dev_hierarchical(); 
            break;
        }
        free(elements[i]); 
      }
    }
    free(elements_type);
  }

  __host__ void set_element(struct dev_hierarchical *matrix, int x, int y) 
  {
    elements[y * nx + x] = (void *) matrix;
    elements_type[y * nx + x] = hierarchical;
  }

  __host__ void set_element(struct dev_dense <matrixEntriesT> *matrix, int x, int y)
  {
    elements[y * nx + x] = (void *) matrix;
    elements_type[y * nx + x] = dense;
  }

  __host__ void print(const int level = 0)
  {
    printf("-- %d x %d nodes. --\n", ny, nx);
    for (int y = 0; y < ny; y++)
    {
      for (int x = 0; x < nx; x++)
      {
        if (elements[y * nx + x] != nullptr) 
        { 
          printf("-- level %d: (%d, %d): --\n", level, y, x);
          switch (elements_type[y * nx + x])
          {
            case empty:
              printf("empty");
              break;
            case dense: 
              ((struct dev_dense <matrixEntriesT> *) elements[y * nx + x]) -> print(); 
              break;
            case low_rank: 
              //TODO 
              break;
            case hierarchical: 
              ((struct dev_hierarchical <matrixEntriesT> *) elements[y * nx + x]) -> print(level + 1); 
              break;
          }
          printf("\n");
        }
      }
    }
  }

  __host__ void loadTestMatrix (const int levels = 1, const int dim = 2)
  {
    for (int y = 0; y < ny; y++)
    {
      for (int x = 0; x < nx; x++)
      {
        if (x == y && levels > 0)
        { 
          struct dev_hierarchical <matrixEntriesT> *e = new dev_hierarchical <matrixEntriesT> (dim, dim);
          e -> loadTestMatrix(levels - 1, dim); 
          elements[y * nx + x] = e;
          elements_type[y * nx + x] = hierarchical;
        }
        else
        {
          struct dev_dense <matrixEntriesT> *e = new dev_dense <matrixEntriesT> (4, 4);
          e -> loadTestMatrix();
          elements[y * nx + x] = e;
          elements_type[y * nx + x] = dense;
        }
      }
    }
  }

};

#endif