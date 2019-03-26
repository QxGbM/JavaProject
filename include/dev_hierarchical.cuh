
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
        if (rows != rows_x) 
        { printf("-- Unmatched Dimensions: (%d, %d) with (%d, 0). --\n\n", y, x, y); return false; }
      }
    }

    for (int x = 0; x < nx; x++)
    {
      const int cols = elements[x].getNx();
      for (int y = 1; y < ny; y++)
      {
        const int cols_y = elements[y * nx + x].getNx();
        if (cols != cols_y)
        { printf("-- Unmatched Dimensions: (%d, %d) with (0, %d). --\n\n", y, x, x); return false; }
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

  __host__ dev_dense <T> * convertToDense() const
  {
    const int nx_d = getNx(), ny_d = getNy();
    if (nx_d > 0 && ny_d > 0)
    {
      dev_dense <T> * d = new dev_dense <T> (nx_d, ny_d);
      for (int y = 0, row = 0; y < ny; y++)
      {
        int rows;
        for (int x = 0, col = 0; x < nx; x++)
        {
          const dev_dense <T> * e = elements[y * nx + x].convertToDense();
          rows = e -> getNy(); int cols = e -> getNx();
          d -> loadArray(e -> getElements(), cols, rows, cols, col, row);
          col += cols;
          delete e;
        }
        row += rows;
      }
      return d;
    }
    else
    {
      return nullptr;
    }
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

  __host__ const h_ops_tree * generateOps_GETRF() const
  {
    const h_index * root = getRootIndex();
    h_ops_tree * ops = new h_ops_tree( new h_ops(getrf, root, getNx(), getNy(), 0) ), * tree = generateOps_GETRF(root);
    ops -> hookup_child(tree);
    delete root;
    return ops;
  }

  __host__ h_ops_tree * generateOps_GETRF (const h_index *self) const
  {
    h_ops_tree * ops = nullptr;
    for (int i = 0; i < nx && i < ny; i++)
    {
      const h_index * index_i = self -> child(i * nx + i);
      h_ops_tree * ops_i = elements[i * nx + i].generateOps_GETRF(index_i);

      for (int j = i + 1; j < nx; j++)
      {
        const h_index * index_j = self -> child(i * nx + j);
        h_ops_tree * ops_j = elements[i * nx + i].generateOps_TRSML(index_i, &elements[i * nx + j], index_j);
        delete index_j;
        ops_i -> hookup_next(ops_j);
      }

      for (int j = i + 1; j < ny; j++)
      {
        const h_index * index_j = self -> child(j * nx + i);
        h_ops_tree * ops_j = elements[i * nx + i].generateOps_TRSMR(index_i, &elements[j * nx + i], index_j);
        delete index_j;
        ops_i -> hookup_next(ops_j);
      }

      delete index_i;

      for (int j = i + 1; j < ny; j++)
      {
        for (int k = i + 1; k < nx; k++)
        {
          const h_index * index_j = self -> child(j * nx + i), * index_k = self -> child(i * nx + k), * index_m = self -> child(j * nx + k);
          h_ops_tree * ops_m = elements[j * nx + k].generateOps_GEMM(index_m, &elements[j * nx + i], index_j, &elements[i * nx + k], index_k);
          delete index_j, index_k, index_m;
          ops_i -> hookup_next(ops_m);
        }
      }

      if (ops == nullptr)
      { ops = ops_i; }
      else
      { ops -> hookup_next (ops_i); }
    }
    return ops;
  }

  /*__host__ dev_h_element <T> * lookup (const int i) const
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
  }*/

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