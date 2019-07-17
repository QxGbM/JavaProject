
#pragma once
#ifndef _DEV_HIERARCHICAL_CUH
#define _DEV_HIERARCHICAL_CUH

#include <pspl.cuh>

template <class T> class dev_hierarchical 
{
private:

  int nx;
  int * x_offsets;

  int ny;
  int * y_offsets;

  dev_h_element <T> * elements;

public:
  
  __host__ dev_hierarchical (const int nx_in, const int ny_in, element_t type = empty, void ** elements_in = nullptr)
  {
    nx = nx_in > 0 ? nx_in : 1;
    x_offsets = new int [nx + 1];

    ny = ny_in > 0 ? ny_in : 1;
    y_offsets = new int [ny + 1];

    elements = new dev_h_element <T> [nx * ny];
    for (int y = 0; y < ny; y++) for (int x = 0; x < nx; x++)
    { setElement((type == empty && elements_in == nullptr) ? nullptr : elements_in[y * nx + x], type, x, y); }

    updateOffsets();
  }

  __host__ ~dev_hierarchical ()
  {
    delete[] x_offsets;
    delete[] y_offsets;
    delete[] elements;
  }

  __host__ inline int getNx_blocks () const
  { return nx; }

  __host__ inline int getNy_blocks () const
  { return ny; }

  __host__ inline int getNx_abs () const
  { return x_offsets[nx]; }

  __host__ inline int getNy_abs () const
  { return y_offsets[ny]; }

  __host__ bool updateOffsets ()
  {
    int accum = 0;
    for (int y = 0; y < ny; y++)
    { y_offsets[y] = accum; accum += elements[y * nx].getNy(); }
    y_offsets[ny] = accum; 
    
    accum = 0;
    for (int x = 0; x < nx; x++)
    { x_offsets[x] = accum; accum += elements[x].getNx(); }
    x_offsets[nx] = accum;

    for (int y = 1; y < ny; y++) for (int x = 1; x < nx; x++)
    {
      const int nx_i = elements[y * nx + x].getNx(), ny_i = elements[y * nx + x].getNy();
      if ((nx_i != x_offsets[x + 1] - x_offsets[x]) && (ny_i != y_offsets[y + 1] - y_offsets[y]))
      { return false; }
    }
    return true;
  }

  __host__ void setElement (void * M, const element_t type, const int x, const int y) 
  {
    if (x < nx && y < ny)
    { elements[y * nx + x].setElement(M, type); }
  }

  __host__ dev_h_element <T> * getElement_blocks (const int y, const int x) const
  { return (x < nx && y < ny) ? &elements[y * nx + x] : nullptr; }

  __host__ T getElement_abs (const int y_in, const int x_in) const
  {
    int block_y, block_x, offset_y = y_in, offset_x = x_in;

    getElement_loc(&offset_y, &offset_x, &block_y, &block_x);

    if (block_y >= 0 && block_x >= 0)
    { return elements[block_y * nx + block_x].getElement(offset_y, offset_x); }
    else
    { return 0; }
  }

  __host__ void getElement_loc (int * offset_y, int * offset_x, int * block_y, int * block_x) const
  {
    int y = 0, x = 0, y_in = * offset_y, x_in = * offset_x;
    while (y < ny && y_in >= y_offsets[y + 1]) { y++; }
    while (x < nx && x_in >= x_offsets[x + 1]) { x++; }

    if (y < ny && x < nx)
    { * offset_y = y_in - y_offsets[y]; * offset_x = x_in - x_offsets[x]; * block_y = y; * block_x = x; }
    else
    { * block_y = -1; * block_x = -1; }
  }

  __host__ void getOffsets_x (int ** x) const
  {
    *x = new int[nx + 1];
    for (int i = 0; i <= nx; i++)
    { (*x)[i] = x_offsets[i]; }
  }

  __host__ void getOffsets_y (int ** y) const
  {
    *y = new int [ny + 1];
    for (int i = 0; i <= ny; i++)
    { (*y)[i] = y_offsets[i]; }
  }

  __host__ dev_dense <T> * convertToDense() const
  {
    const int nx_d = getNx_abs(), ny_d = getNy_abs();
    if (nx_d > 0 && ny_d > 0)
    {
      dev_dense <T> * d = new dev_dense <T> (nx_d, ny_d);
      T * d_elements = d -> getElements();
      for (int y = 0; y < ny_d; y++) for (int x = 0; x < nx_d; x++)
      { d_elements[y * nx_d + x] = getElement_abs (y, x); }
      return d;
    }
    else
    { return nullptr; }
  }

  __host__ inline h_index * getRootIndex () const
  { return new h_index (this); }

  __host__ h_ops_tree * generateOps_GETRF (const h_index * self, dev_temp * tmp_mngr) const
  {
    h_ops_tree * op = new h_ops_tree (getrf, self);

    int n = nx > ny ? ny : nx, * child_offset = new int[n + 1];
    child_offset[0] = 0;

    for (int i = 1; i <= n; i++)
    { child_offset[i] = child_offset[i - 1] + (nx - i + 1) * (ny - i + 1); }

    op -> resizeChildren(child_offset[n]);

#pragma omp parallel for if (omp_in_parallel() == 0)
    for (int i = 0; i < n; i++)
    {
      const h_index index_i = h_index (this, self, i, i);
      h_ops_tree * op_i = elements[i * nx + i].generateOps_GETRF(&index_i, tmp_mngr);
      op -> setChild(op_i, child_offset[i]);
      delete op_i;
      const int rows = ny - i - 1, cols = nx - i - 1;

      for (int j = i + 1; j < nx; j++)
      {
        const h_index index_j = h_index (this, self, i, j);
        h_ops_tree * op_j = elements[i * nx + i].generateOps_TRSML(&index_i, &elements[i * nx + j], &index_j, tmp_mngr);
        op -> setChild(op_j, child_offset[i] + j - i);
        delete op_j;
      }

      for (int j = i + 1; j < ny; j++)
      {
        const h_index index_j = h_index (this, self, j, i);
        h_ops_tree * op_j = elements[i * nx + i].generateOps_TRSMR(&index_i, &elements[j * nx + i], &index_j, tmp_mngr);
        op -> setChild(op_j, child_offset[i] + cols + j - i);
        delete op_j;
      }

      for (int j = 0; j < rows * cols; j++)
      {
        const int row = j / cols + i + 1, col = j - (row - i - 1) * cols + i + 1;
        const h_index index_j = h_index (this, self, row, i), index_k = h_index (this, self, i, col), index_m = h_index (this, self, row, col);
        h_ops_tree * op_j = elements[row * nx + col].generateOps_GEMM(&index_m, &elements[row * nx + i], &index_j, &elements[i * nx + col], &index_k, tmp_mngr);
        op -> setChild(op_j, child_offset[i] + rows + cols + j + 1);
        delete op_j;
      }
    }

    delete[] child_offset;
    return op;
  }

  __host__ h_ops_tree * generateOps_TRSML (const h_index * self, const dev_dense <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    h_ops_tree * op = new h_ops_tree (trsml, index_b, self);

    int n = nx > ny ? ny : nx, * child_offset = new int[n + 1];
    child_offset[0] = 0;

    for (int i = 1; i <= n; i++)
    { child_offset[i] = child_offset[i - 1] + ny - i + 1; }

    op -> resizeChildren(child_offset[n]);

#pragma omp parallel for if (omp_in_parallel() == 0)
    for (int i = 0; i < n; i++)
    {
      h_index index_i = h_index (this, self, i, i), index_bi = h_index (index_b, y_offsets[i], 0, index_i.getNy(), index_b -> getNx());
      h_ops_tree * op_i = elements[i * nx + i].generateOps_TRSML(&index_i, B, &index_bi, tmp_mngr);
      op -> setChild(op_i, child_offset[i]);
      delete op_i;

      for (int j = i + 1; j < ny; j++)
      {
        h_index index_j = h_index (this, self, j, i), index_bj = h_index (index_b, y_offsets[j], 0, index_j.getNy(), index_b -> getNx());
        h_ops_tree * op_j = B -> generateOps_GEMM(&index_bj, &elements[j * nx + i], &index_j, B, &index_bi, tmp_mngr);
        op -> setChild(op_j, child_offset[i] + j - i);
        delete op_j;
      }
    }

    delete[] child_offset;
    return op;
  }

  __host__ h_ops_tree * generateOps_TRSML (const h_index * self, const dev_low_rank <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    h_ops_tree * op = new h_ops_tree (trsml, index_b, self);

    int n = nx > ny ? ny : nx, * child_offset = new int[n + 1];
    child_offset[0] = 0;

    for (int i = 1; i <= n; i++)
    { child_offset[i] = child_offset[i - 1] + ny - i + 1; }

    op -> resizeChildren(child_offset[n]);

#pragma omp parallel for if (omp_in_parallel() == 0)
    for (int i = 0; i < n; i++)
    {
      h_index index_i = h_index (this, self, i, i), index_biu = h_index (index_b, y_offsets[i], 0, index_i.getNy(), index_b -> getNx());
      index_biu.setU();
      h_ops_tree * op_i = elements[i * nx + i].generateOps_TRSML(&index_i, B, &index_biu, tmp_mngr);
      op -> setChild(op_i, child_offset[i]);
      delete op_i;

      for (int j = i + 1; j < ny; j++)
      {
        h_index index_j = h_index (this, self, j, i), index_bju = h_index (index_b, y_offsets[j], 0, index_j.getNy(), index_b -> getNx());
        index_bju.setU();
        h_ops_tree * op_j = B -> generateOps_GEMM(&index_bju, &elements[j * nx + i], &index_j, B, &index_biu, tmp_mngr);
        op -> setChild(op_j, child_offset[i] + j - i);
        delete op_j;
      }
    }

    delete[] child_offset;
    return op;
  }

  __host__ h_ops_tree * generateOps_TRSML (const h_index * self, const dev_hierarchical <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    if (ny != B -> ny) 
    { printf("Matrices are partitioned differently in H-H TRSML.\n"); return nullptr; }

    h_ops_tree * op = new h_ops_tree (trsml, index_b, self);

    int n = nx > ny ? ny : nx, * child_offset = new int[n + 1];
    child_offset[0] = 0;

    for (int i = 1; i <= n; i++)
    { child_offset[i] = child_offset[i - 1] + (B -> nx) * (ny - i + 1); }

    op -> resizeChildren(child_offset[n]);

#pragma omp parallel for if (omp_in_parallel() == 0)
    for (int i = 0; i < n; i++)
    {
      h_index index_i = h_index (this, self, i, i);

      for (int j = 0; j < B -> nx; j++)
      {
        h_index index_bj = h_index (B, index_b, i, j);

        h_ops_tree * op_j = elements[i * nx + i].generateOps_TRSML(&index_i, &(B -> elements)[i * (B -> nx) + j], &index_bj, tmp_mngr);
        op -> setChild(op_j, child_offset[i] + j);
        delete op_j;

        for (int k = i + 1; k < ny; k++)
        {
          h_index index_k = h_index (this, self, k, i), index_bk = h_index (B, index_b, k, j);
          h_ops_tree * op_k = (B -> elements[k * (B -> nx) + j]).generateOps_GEMM(&index_bk, &elements[k * nx + i], &index_k, &(B -> elements)[i * (B -> nx) + j], &index_bj, tmp_mngr);
          op -> setChild(op_k, child_offset[i] + (k - i) * B -> nx + j);
          delete op_k;
        }
      }
    }

    delete[] child_offset;
    return op;
  }

  __host__ h_ops_tree * generateOps_TRSML (const h_index * self, const dev_h_element <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    const dev_hierarchical <T> *h_b = B -> getElementHierarchical();
    const dev_low_rank <T> *lr_b = B -> getElementLowRank();
    const dev_dense <T> *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_TRSML (self, d_b, index_b, tmp_mngr); }
    if (lr_b != nullptr)
    { return generateOps_TRSML (self, lr_b, index_b, tmp_mngr); }
    if (h_b != nullptr)
    { return generateOps_TRSML (self, h_b, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_TRSMR (const h_index * self, const dev_dense <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    h_ops_tree * op = new h_ops_tree (trsmr, index_b, self);

    int n = nx > ny ? ny : nx, * child_offset = new int[n + 1];
    child_offset[0] = 0;

    for (int i = 1; i <= n; i++)
    { child_offset[i] = child_offset[i - 1] + nx - i + 1; }

    op -> resizeChildren(child_offset[n]);

#pragma omp parallel for if (omp_in_parallel() == 0)
    for (int i = 0; i < n; i++)
    {
      h_index index_i = h_index (this, self, i, i), index_bi = h_index (index_b, 0, x_offsets[i], index_b -> getNy(), index_i.getNx());
      h_ops_tree * op_i = elements[i * nx + i].generateOps_TRSMR(&index_i, B, &index_bi, tmp_mngr);
      op -> setChild(op_i, child_offset[i]);
      delete op_i;

      for (int j = i + 1; j < nx; j++)
      {
        h_index index_j = h_index (this, self, i, j), index_bj = h_index (index_b, 0, x_offsets[j], index_b -> getNy(), index_j.getNx());
        h_ops_tree * op_j = B -> generateOps_GEMM(&index_bj, &elements[j * nx + i], &index_j, B, &index_bi, tmp_mngr);
        op -> setChild(op_j, child_offset[i] + j - i);
        delete op_j;
      }
    }

    delete[] child_offset;
    return op;
  }

  __host__ h_ops_tree * generateOps_TRSMR (const h_index * self, const dev_low_rank <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    h_ops_tree * op = new h_ops_tree (trsmr, index_b, self);

    int n = nx > ny ? ny : nx, * child_offset = new int[n + 1];
    child_offset[0] = 0;

    for (int i = 1; i <= n; i++)
    { child_offset[i] = child_offset[i - 1] + nx - i + 1; }

    op -> resizeChildren(child_offset[n]);

#pragma omp parallel for if (omp_in_parallel() == 0)
    for (int i = 0; i < n; i++)
    {
      h_index index_i = h_index (this, self, i, i), index_biv = h_index (index_b, 0, x_offsets[i], index_b -> getNy(), index_i.getNx());
      index_biv.setVT();
      h_ops_tree * op_i = elements[i * nx + i].generateOps_TRSMR(&index_i, B, &index_biv, tmp_mngr);
      op -> setChild(op_i, child_offset[i]);
      delete op_i;

      for (int j = i + 1; j < nx; j++)
      {
        h_index index_j = h_index (this, self, i, j), index_bjv = h_index (index_b, 0, x_offsets[j], index_b -> getNy(), index_j.getNx());
        index_bjv.setVT();
        h_ops_tree * op_j = B -> generateOps_GEMM(&index_bjv, B, &index_biv, &elements[i * nx + j], &index_j, tmp_mngr);
        op -> setChild(op_j, child_offset[i] + j - i);
        delete op_j;
      }
    }

    delete[] child_offset;
    return op;
  }

  __host__ h_ops_tree * generateOps_TRSMR (const h_index * self, const dev_hierarchical <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    if (nx != B -> nx)
    { printf("Matrices are partitioned differently in H-H TRSMR.\n"); return nullptr; }

    h_ops_tree * op = new h_ops_tree (trsmr, index_b, self);

    int n = nx > ny ? ny : nx, * child_offset = new int[n + 1];
    child_offset[0] = 0;

    for (int i = 1; i <= n; i++)
    { child_offset[i] = child_offset[i - 1] + (B -> ny) * (nx - i + 1); }

    op -> resizeChildren(child_offset[n]);

#pragma omp parallel for if (omp_in_parallel() == 0)
    for (int i = 0; i < n; i++)
    {
      h_index index_i = h_index (this, self, i, i);

      for (int j = 0; j < B -> ny; j++)
      {
        h_index index_bj = h_index (B, index_b, j, i);

        h_ops_tree * op_j = elements[i * nx + i].generateOps_TRSMR(&index_i, &(B -> elements)[j * (B -> nx) + i], &index_bj, tmp_mngr);
        op -> setChild(op_j, child_offset[i] + j);
        delete op_j;

        for (int k = i + 1; k < nx; k++)
        {
          h_index index_k = h_index (this, self, i, k), index_bk = h_index (B, index_b, j, k);
          h_ops_tree * op_k = (B -> elements[j * (B -> nx) + k]).generateOps_GEMM(&index_bk, &(B -> elements)[j * (B -> nx) + i], &index_bj, &elements[i * nx + k], &index_k, tmp_mngr);
          op -> setChild(op_k, child_offset[i] + (k - i) * B -> ny + j);
          delete op_k;
        }
      }
    }

    delete[] child_offset;
    return op;
  }

  __host__ h_ops_tree * generateOps_TRSMR (const h_index * self, const dev_h_element <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    const dev_hierarchical <T> *h_b = B -> getElementHierarchical();
    const dev_low_rank <T> *lr_b = B -> getElementLowRank();
    const dev_dense <T> *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_TRSMR (self, d_b, index_b, tmp_mngr); }
    if (lr_b != nullptr)
    { return generateOps_TRSMR (self, lr_b, index_b, tmp_mngr); }
    if (h_b != nullptr)
    { return generateOps_TRSMR (self, h_b, index_b, tmp_mngr); }

    return nullptr;  
  }

  __host__ h_ops_tree * generateOps_ACCM (const h_index * self, const h_index * index_tmp_lr) const
  {
    h_ops_tree * op = new h_ops_tree (accum, self, index_tmp_lr);
    op -> resizeChildren(nx * ny);

#pragma omp parallel for if (omp_in_parallel() == 0)
    for (int i = 0; i < ny * nx; i++)
    {
      const int row = i / nx, col = i - row * nx;
      const h_index index_m = h_index (this, self, row, col), index_lr = h_index (index_tmp_lr, y_offsets[row], x_offsets[col], index_m.getNy(), index_m.getNx());
      h_ops_tree * op_i = elements[i].generateOps_ACCM(&index_m, &index_lr);
      op -> setChild(op_i, i);
      delete op_i;
    }

    return op;  
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index * self, const dev_dense <T> * A, const h_index * index_a, const dev_dense <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b);
    op -> resizeChildren(nx * ny);

    const int k = index_a -> getNx(index_b -> getNy());

#pragma omp parallel for if (omp_in_parallel() == 0)
    for (int i = 0; i < ny * nx; i++)
    {
      const int row = i / nx, col = i - row * nx;
      const h_index index_m = h_index (this, self, row, col), index_ai = h_index (index_a, y_offsets[row], 0, index_m.getNy(), k), index_bj = h_index (index_b, 0, x_offsets[col], k, index_m.getNx());
      h_ops_tree * op_i = elements[i].generateOps_GEMM(&index_m, A, &index_ai, B, &index_bj, tmp_mngr);
      op -> setChild(op_i, i);
      delete op_i;
    }

    return op;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index * self, const dev_low_rank <T> * A, const h_index * index_a, const dev_dense <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b), * op_;
    op -> resizeChildren (2);

    int rank_a = index_a -> getRank(), tmp_size = rank_a * index_b -> getNx(), block_id;
#pragma omp critical
    { block_id = tmp_mngr -> requestTemp(tmp_size); }

    h_index index_tmp = h_index (self), index_av = h_index (index_a);
    index_tmp.setTemp_Low_Rank (block_id, rank_a);
    index_tmp.setU_data (index_a);

    op_ = generateOps_ACCM (self, &index_tmp);
    op -> setChild(op_, 1);
    delete op_;

    index_tmp.setVT();
    index_av.setVT();

    op_ = new h_ops_tree (gemm, &index_tmp, &index_av, index_b);
    op -> setChild (op_, 0);
    delete op_;

    return op;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index * self, const dev_hierarchical <T> * A, const h_index * index_a, const dev_dense <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    if (ny != A -> ny)
    { printf("Matrices are partitioned differently in H-H.D GEMM.\n"); return nullptr; }

    h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b);
    op -> resizeChildren(nx * ny * A -> nx);

#pragma omp parallel for if (omp_in_parallel() == 0)
    for (int i = 0; i < ny * nx; i++)
    {
      const int row = i / nx, col = i - row * nx;
      const h_index index_m = h_index (this, self, row, col);
      for (int k = 0; k < A -> nx; k++)
      {
        const h_index index_ak = h_index (A, index_a, row, k), index_bk = h_index (index_b, (A -> x_offsets)[k], x_offsets[col], index_ak.getNx(), index_m.getNx());
        h_ops_tree * op_k = elements[i].generateOps_GEMM(&index_m, &(A -> elements[row * (A -> nx) + k]), &index_ak, B, &index_bk, tmp_mngr);
        op -> setChild(op_k, i * (A -> nx) + k);
        delete op_k;
      }
    }

    return op;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index * self, const dev_h_element <T> * A, const h_index * index_a, const dev_dense <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    const dev_hierarchical <T> *h_a = A -> getElementHierarchical();
    const dev_low_rank <T> *lr_a = A -> getElementLowRank();
    const dev_dense <T> *d_a = A -> getElementDense();

    if (d_a != nullptr)
    { return generateOps_GEMM (self, d_a, index_a, B, index_b, tmp_mngr); }
    if (lr_a != nullptr)
    { return generateOps_GEMM (self, lr_a, index_a, B, index_b, tmp_mngr); }
    if (h_a != nullptr)
    { return generateOps_GEMM (self, h_a, index_a, B, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index * self, const dev_dense <T> * A, const h_index * index_a, const dev_low_rank <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b), * op_;

    op -> resizeChildren (2);

    int rank_b = index_b -> getRank(), tmp_size = rank_b * index_a -> getNy(), block_id;
#pragma omp critical
    { block_id = tmp_mngr -> requestTemp(tmp_size); }

    h_index index_tmp = h_index (self), index_bu = h_index (index_b);
    index_tmp.setTemp_Low_Rank (block_id, rank_b);
    index_tmp.setVT_data (index_b);

    op_ = generateOps_ACCM (self, &index_tmp);
    op -> setChild (op_, 1);
    delete op_;

    index_tmp.setU();
    index_bu.setU();

    op_ = new h_ops_tree (gemm, &index_tmp, index_a, &index_bu);
    op -> setChild (op_, 0);
    delete op_;

    return op;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index * self, const dev_low_rank <T> * A, const h_index * index_a, const dev_low_rank <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b), * op_;

    op -> resizeChildren (2);

    h_index index_tmp = h_index (self);
    bool a; 
    int rank = index_a -> getMinRank (index_b, &a), tmp_size = rank * (a ? index_b -> getNx() : index_a -> getNy()), block_id;

#pragma omp critical
    { block_id = tmp_mngr -> requestTemp(tmp_size); }

    index_tmp.setTemp_Low_Rank(block_id, rank);
    if (a)
    { index_tmp.setU_data(index_a); }
    else
    { index_tmp.setVT_data(index_b); }

    op_ = generateOps_ACCM (self, &index_tmp);
    op -> setChild (op_, 1);
    delete op_;

    if (a)
    {
      h_index index_av = h_index (index_a);
      index_tmp.setVT();
      index_av.setVT();

      op_ = new h_ops_tree (gemm, &index_tmp, &index_av, index_b);
    }
    else
    {
      h_index index_bu = h_index (index_b);
      index_tmp.setU();
      index_bu.setU();

      op_ = new h_ops_tree (gemm, &index_tmp, index_a, &index_bu);
    }

    op -> setChild (op_, 0);
    delete op_;

    return op;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index * self, const dev_hierarchical <T> * A, const h_index * index_a, const dev_low_rank <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    if (ny != A -> ny)
    { printf("Matrices are partitioned differently in H-H.LR GEMM.\n"); return nullptr; }

    h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b);

    op -> resizeChildren (2);

    int rank_b = index_b -> getRank(), tmp_size = rank_b * index_a -> getNy(self -> getNy()), block_id;

#pragma omp critical
    { block_id = tmp_mngr -> requestTemp(tmp_size); }

    h_index index_tmp = h_index (self), index_bu = h_index (index_b);
    index_tmp.setTemp_Low_Rank (block_id, rank_b);
    index_tmp.setVT_data (index_b);

    h_ops_tree * op_ = generateOps_ACCM (self, &index_tmp);
    op -> setChild (op_, 1);
    delete op_;

    index_tmp.setU();
    index_bu.setU();

    op_ = dev_low_rank<T>::generateOps_GEMM (&index_tmp, A, index_a, B, &index_bu, tmp_mngr);
    op -> setChild (op_, 0);
    delete op_;

    return op;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index * self, const dev_h_element <T> * A, const h_index *index_a, const dev_low_rank <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    const dev_hierarchical <T> *h_a = A -> getElementHierarchical();
    const dev_low_rank <T> *lr_a = A -> getElementLowRank();
    const dev_dense <T> *d_a = A -> getElementDense();

    if (d_a != nullptr)
    { return generateOps_GEMM (self, d_a, index_a, B, index_b, tmp_mngr); }
    if (lr_a != nullptr)
    { return generateOps_GEMM (self, lr_a, index_a, B, index_b, tmp_mngr); }
    if (h_a != nullptr)
    { return generateOps_GEMM (self, h_a, index_a, B, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index * self, const dev_dense <T> * A, const h_index * index_a, const dev_hierarchical <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    if (nx != B -> nx)
    { printf("Matrices are partitioned differently in H-D.H GEMM.\n"); return nullptr; }

    h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b);
    op -> resizeChildren (nx * ny * B -> ny);

#pragma omp parallel for if (omp_in_parallel() == 0)
    for (int i = 0; i < ny * nx; i++)
    {
      const int row = i / nx, col = i - row * nx;
      const h_index index_m = h_index (this, self, row, col);
      for (int k = 0; k < B -> ny; k++)
      {
        const h_index index_bk = h_index (B, index_b, k, col), index_ak = h_index (index_a, y_offsets[row], (B -> y_offsets)[k], index_m.getNy(), index_bk.getNy());
        h_ops_tree * op_k = elements[i].generateOps_GEMM(&index_m, A, &index_ak, &(B -> elements[k * (B -> nx) + col]), &index_bk, tmp_mngr);
        op -> setChild(op_k, i * (B -> ny) + k);
        delete op_k;
      }
    }

    return op;  
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index * self, const dev_low_rank <T> * A, const h_index * index_a, const dev_hierarchical <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    if (nx != B -> nx)
    { printf("Matrices are partitioned differently in H-LR.H GEMM.\n"); return nullptr; }

    h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b);
    op -> resizeChildren (2);

    int rank_a = index_a -> getRank(), tmp_size = rank_a * index_b -> getNx(self -> getNx()), block_id;
#pragma omp critical
    { block_id = tmp_mngr -> requestTemp(tmp_size); }

    h_index index_tmp = h_index (self), index_av = h_index (index_a);
    index_tmp.setTemp_Low_Rank (block_id, rank_a);
    index_tmp.setU_data (index_a);

    h_ops_tree * op_ = generateOps_ACCM (self, &index_tmp);
    op -> setChild(op_, 1);
    delete op_;

    index_tmp.setVT();
    index_av.setVT();

    op_ = dev_low_rank<T>::generateOps_GEMM (&index_tmp, A, &index_av, B, index_b, tmp_mngr);
    op -> setChild (op_, 0);
    delete op_;

    return op;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index * self, const dev_hierarchical <T> * A, const h_index * index_a, const dev_hierarchical <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    if (ny != A -> ny || nx != B -> nx || A -> nx != B -> ny)
    { printf("Partition error in H-H.H GEMM.\n"); return nullptr; }

    h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b);
    op -> resizeChildren(ny * nx * A -> nx);

#pragma omp parallel for if (omp_in_parallel() == 0)
    for (int i = 0; i < ny * nx; i++)
    {
      const int row = i / nx, col = i - row * nx;
      const h_index index_m = h_index (this, self, row, col);

      for (int k = 0; k < A -> nx; k++)
      {
        const h_index index_ak = h_index (A, index_a, row, k), index_bk = h_index (B, index_b, k, col);

        h_ops_tree * op_k = elements[i].generateOps_GEMM (&index_m, &(A -> elements)[row * (A -> nx) + k], &index_ak, &(B -> elements)[k * (B -> nx) + col], &index_bk, tmp_mngr);
        op -> setChild(op_k, i * A -> nx + k);
        delete op_k;
      }
    }

    return op;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index * self, const dev_h_element <T> * A, const h_index * index_a, const dev_hierarchical <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    const dev_hierarchical <T> *h_a = A -> getElementHierarchical();
    const dev_low_rank <T> *lr_a = A -> getElementLowRank();
    const dev_dense <T> *d_a = A -> getElementDense();

    if (d_a != nullptr)
    { return generateOps_GEMM (self, d_a, index_a, B, index_b, tmp_mngr); }
    if (lr_a != nullptr)
    { return generateOps_GEMM (self, lr_a, index_a, B, index_b, tmp_mngr); }
    if (h_a != nullptr)
    { return generateOps_GEMM (self, h_a, index_a, B, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index * self, const dev_dense <T> * A, const h_index * index_a, const dev_h_element <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    const dev_hierarchical <T> *h_b = B -> getElementHierarchical();
    const dev_low_rank <T> *lr_b = B -> getElementLowRank();
    const dev_dense <T> *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, d_b, index_b, tmp_mngr); }
    if (lr_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, lr_b, index_b, tmp_mngr); }
    if (h_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, h_b, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index * self, const dev_low_rank <T> * A, const h_index * index_a, const dev_h_element <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    const dev_hierarchical <T> *h_b = B -> getElementHierarchical();
    const dev_low_rank <T> *lr_b = B -> getElementLowRank();
    const dev_dense <T> *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, d_b, index_b, tmp_mngr); }
    if (lr_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, lr_b, index_b, tmp_mngr); }
    if (h_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, h_b, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index * self, const dev_hierarchical <T> * A, const h_index * index_a, const dev_h_element <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    const dev_hierarchical <T> *h_b = B -> getElementHierarchical();
    const dev_low_rank <T> *lr_b = B -> getElementLowRank();
    const dev_dense <T> *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, d_b, index_b, tmp_mngr); }
    if (lr_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, lr_b, index_b, tmp_mngr); }
    if (h_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, h_b, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index * self, const dev_h_element <T> * A, const h_index * index_a, const dev_h_element <T> * B, const h_index * index_b, dev_temp * tmp_mngr) const
  {
    const dev_hierarchical <T> *h_b = B -> getElementHierarchical();
    const dev_low_rank <T> *lr_b = B -> getElementLowRank();
    const dev_dense <T> *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, d_b, index_b, tmp_mngr); }
    if (lr_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, lr_b, index_b, tmp_mngr); }
    if (h_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, h_b, index_b, tmp_mngr); }

    return nullptr;
  }

  __host__ cudaError_t loadBinary (FILE * stream, const bool reverse_bytes = true)
  {
    cudaError_t error = cudaSuccess;
    for (int i = 0; i < nx * ny; i++)
    {
      if (error != cudaSuccess)
      { return error; }
      else
      { elements[i].loadBinary (stream, reverse_bytes); }
    }
    return error;
  }

  __host__ static dev_hierarchical <T> * readStructureFromFile (FILE * stream, const int shadow_rank = _DEFAULT_SHADOW_RANK)
  {
    element_t type;
    void * h = dev_h_element <T> :: readStructureFromFile(stream, &type, shadow_rank);

    if (type == hierarchical)
    { return (dev_hierarchical <T> *) h; }
    else
    {
      printf("The Matrix Loaded is not a hierarchical matrix.\n");

      if (type == dense)
      { dev_dense<T> * d = (dev_dense<T> *) h; delete d; }
      else if (type == low_rank)
      { dev_low_rank<T> * lr = (dev_low_rank<T> *) h; delete lr; }

      return nullptr; 
    }

  }

  __host__ static dev_hierarchical <T> * readFromFile (const char * file_name, const int shadow_rank = _DEFAULT_SHADOW_RANK, const bool reverse_bytes = true)
  {
    char str[32], bin[32];
    strcpy(str, file_name); strcat(str, ".struct");
    strcpy(bin, file_name); strcat(bin, ".bin");

    FILE * stream = fopen(str, "r");
    dev_hierarchical <T> * a = dev_hierarchical <T> :: readStructureFromFile (stream, shadow_rank);
    fclose(stream);

    if (a != nullptr)
    {
      stream = fopen(bin, "rb");
      a -> loadBinary(stream, reverse_bytes);
      fclose(stream);
    }

    return a;
  }

  __host__ void print(const h_index * index_in) const
  {
    for (int i = 0; i < ny * nx; i++)
    {
      const int row = i / nx, col = i - row * nx;
      const h_index * i_index = new h_index(this, index_in, row, col);
      elements[i].print(i_index);
      delete i_index;
    }
  }

  __host__ void print(const int level, const int * indexes) const
  {
    const int i = indexes[0];
    if (level > 1)
    { elements[i].getElementHierarchical() -> print(level - 1, &indexes[1]); }
    else
    { const h_index * root = getRootIndex(); elements[i].print(root); delete root; }
  }

  __host__ void print() const
  {
    const h_index * root = getRootIndex();
    print(root);
    delete root;
  }


};

#endif