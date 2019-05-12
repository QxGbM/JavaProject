
#ifndef _DEV_DENSE_CUH
#define _DEV_DENSE_CUH

#include <pspl.cuh>

template <class T> class dev_dense 
{
private:
  int device_id;

  int nx;
  int ny;
  int ld;

  T * elements;

  bool pivoted;
  int * pivot;

public:
  __host__ dev_dense (const int nx_in, const int ny_in, const int ld_in = 0, const bool alloc_pivot = false, const int device_id_in = -1)
  {
    nx = nx_in;
    ny = ny_in;
    ld = (nx > ld_in) ? nx : ld_in;

    if (device_id_in >= 0 && cudaSetDevice(device_id_in) == cudaSuccess)
    { device_id = device_id_in; }
    else
    { device_id = 0; }

    if (cudaMallocManaged(&elements, ld * ny * sizeof(T), cudaMemAttachGlobal) == cudaSuccess)
    { cudaMemset(elements, 0, ld * ny * sizeof(T)); }
    else
    { elements = nullptr; }
    
    if (alloc_pivot && cudaMallocManaged(&pivot, ny * sizeof(int), cudaMemAttachGlobal) == cudaSuccess)
    { cudaMemset(pivot, 0, ny * sizeof(int)); pivoted = true; }
    else
    { pivot = nullptr; pivoted = false; }

  }

  __host__ ~dev_dense ()
  {
    cudaFree(elements);
    if (pivoted) { cudaFree(pivot); }
  }

  __host__ inline int getNx () const 
  { return nx; }

  __host__ inline int getNy () const 
  { return ny; }

  __host__ inline int getLd () const 
  { return ld; }

  __host__ inline T * getElements (const int offset = 0) const 
  { return &elements[offset]; }

  __host__ inline int * getPivot (const int offset = 0) const 
  { return pivoted ? &pivot[offset / ld] : nullptr; }

  __host__ cudaError_t resize (const int ld_in, const int ny_in)
  {
    cudaError_t error = resizeColumn(ld_in);
    return error == cudaSuccess ? resizeRow(ny_in) : error;
  }

  __host__ cudaError_t resizeColumn (const int ld_in)
  {
    if (ld_in > 0 && ld_in != ld)
    {
      T * e = nullptr;
      cudaError_t error = cudaMallocManaged (&e, ld_in * ny * sizeof(T), cudaMemAttachGlobal);
      if (error != cudaSuccess) { return error; }

      for (int y = 0; y < ny; y++) for (int x = 0; x < nx && x < ld_in; x++)
      { e[y * ld_in + x] = elements[y * ld + x]; }

      cudaFree(elements);
      ld = ld_in;
      nx = (nx > ld) ? ld : nx;
      elements = e;
    }
    return cudaSuccess;
  }

  __host__ cudaError_t resizeRow (const int ny_in)
  {
    if (ny_in > 0 && ny_in != ny)
    {
      T * e = nullptr;
      cudaError_t error = cudaMallocManaged (&e, ld * ny_in * sizeof(T), cudaMemAttachGlobal);
      if (error != cudaSuccess) { return error; }

      for (int y = 0; y < ny_in && y < ny; y++) for (int x = 0; x < nx; x++)
      { e[y * ld + x] = elements[y * ld + x]; }

      cudaFree(elements);
      elements = e;

      if (pivoted)
      {
        int * p = nullptr;
        error = cudaMallocManaged (&p, ny_in * sizeof(int), cudaMemAttachGlobal);
        if (error != cudaSuccess) { return error; }

        for (int y = 0; y < ny_in && y < ny; y++)
        { p[y] = pivot[y]; }

        cudaFree(pivot);
        pivot = p;
      }

      ny = ny_in;
    }
    return cudaSuccess;
  }

  __host__ h_ops_tree * generateOps_GETRF (const h_index * self) const
  { 
    return new h_ops_tree (getrf_d, self); 
  }

  __host__ h_ops_tree * generateOps_TRSML (const h_index *self, const dev_dense <T> *B, const h_index *index_b) const
  {
    return new h_ops_tree (trsml_d, index_b, self);
  }

  __host__ h_ops_tree * generateOps_TRSML (const h_index *self, const dev_low_rank <T> *B, const h_index *index_b) const
  {
    return new h_ops_tree (trsml_lr, index_b, self);
  }

  __host__ h_ops_tree * generateOps_TRSML (const h_index *self, const dev_hierarchical <T> *B, const h_index *index_b) const
  {
    h_ops_tree * op = new h_ops_tree (trsml_d, index_b, self);

    return op;
  }

  __host__ h_ops_tree * generateOps_TRSML (const h_index *self, const dev_h_element <T> *B, const h_index *index_b) const
  {
    const dev_hierarchical <T> *h_b = B -> getElementHierarchical();
    const dev_low_rank <T> *lr_b = B -> getElementLowRank();
    const dev_dense <T> *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_TRSML (self, d_b, index_b); }
    if (lr_b != nullptr)
    { return generateOps_TRSML (self, lr_b, index_b); }
    if (h_b != nullptr)
    { return generateOps_TRSML (self, h_b, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_TRSMR (const h_index *self, const dev_dense <T> *B, const h_index *index_b) const
  {
    return new h_ops_tree (trsmr_d, index_b, self);
  }

  __host__ h_ops_tree * generateOps_TRSMR (const h_index *self, const dev_low_rank <T> *B, const h_index *index_b) const
  {
    return new h_ops_tree (trsmr_lr, index_b, self);
  }

  __host__ h_ops_tree * generateOps_TRSMR (const h_index *self, const dev_hierarchical <T> *B, const h_index *index_b) const
  {
    return nullptr;
  }

  __host__ h_ops_tree * generateOps_TRSMR (const h_index *self, const dev_h_element <T> *B, const h_index *index_b) const
  {
    const dev_hierarchical <T> *h_b = B -> getElementHierarchical();
    const dev_low_rank <T> *lr_b = B -> getElementLowRank();
    const dev_dense <T> *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_TRSMR (self, d_b, index_b); }
    if (lr_b != nullptr)
    { return generateOps_TRSMR (self, lr_b, index_b); }
    if (h_b != nullptr)
    { return generateOps_TRSMR (self, h_b, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_dense <T> *A, const h_index *index_a, const dev_dense <T> *B, const h_index *index_b) const
  {
    return new h_ops_tree (gemm_d_d_d, self, index_a, index_b); 
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_low_rank <T> *A, const h_index *index_a, const dev_dense <T> *B, const h_index *index_b) const
  {
    return new h_ops_tree (gemm_d_lr_d, self, index_a, index_b); 
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_hierarchical <T> *A, const h_index *index_a, const dev_dense <T> *B, const h_index *index_b) const
  {
    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_h_element <T> *A, const h_index *index_a, const dev_dense <T> *B, const h_index *index_b) const
  {
    const dev_hierarchical <T> *h_a = A -> getElementHierarchical();
    const dev_low_rank <T> *lr_a = A -> getElementLowRank();
    const dev_dense <T> *d_a = A -> getElementDense();

    if (d_a != nullptr)
    { return generateOps_GEMM (self, d_a, index_a, B, index_b); }
    if (lr_a != nullptr)
    { return generateOps_GEMM (self, lr_a, index_a, B, index_b); }
    if (h_a != nullptr)
    { return generateOps_GEMM (self, h_a, index_a, B, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_dense <T> *A, const h_index *index_a, const dev_low_rank <T> *B, const h_index *index_b) const
  {
    return new h_ops_tree (gemm_d_d_lr, self, index_a, index_b);
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_low_rank <T> *A, const h_index *index_a, const dev_low_rank <T> *B, const h_index *index_b) const
  {
    return new h_ops_tree (gemm_d_lr_lr, self, index_a, index_b);
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_hierarchical <T> *A, const h_index *index_a, const dev_low_rank <T> *B, const h_index *index_b) const
  {
    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_h_element <T> *A, const h_index *index_a, const dev_low_rank <T> *B, const h_index *index_b) const
  {
    const dev_hierarchical <T> *h_a = A -> getElementHierarchical();
    const dev_low_rank <T> *lr_a = A -> getElementLowRank();
    const dev_dense <T> *d_a = A -> getElementDense();

    if (d_a != nullptr)
    { return generateOps_GEMM (self, d_a, index_a, B, index_b); }
    if (lr_a != nullptr)
    { return generateOps_GEMM (self, lr_a, index_a, B, index_b); }
    if (h_a != nullptr)
    { return generateOps_GEMM (self, h_a, index_a, B, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_dense <T> *A, const h_index *index_a, const dev_hierarchical <T> *B, const h_index *index_b) const
  {
    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_low_rank <T> *A, const h_index *index_a, const dev_hierarchical <T> *B, const h_index *index_b) const
  {
    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_hierarchical <T> *A, const h_index *index_a, const dev_hierarchical <T> *B, const h_index *index_b) const
  {
    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_h_element <T> *A, const h_index *index_a, const dev_hierarchical <T> *B, const h_index *index_b) const
  {
    const dev_hierarchical <T> *h_a = A -> getElementHierarchical();
    const dev_low_rank <T> *lr_a = A -> getElementLowRank();
    const dev_dense <T> *d_a = A -> getElementDense();

    if (d_a != nullptr)
    { return generateOps_GEMM (self, d_a, index_a, B, index_b); }
    if (lr_a != nullptr)
    { return generateOps_GEMM (self, lr_a, index_a, B, index_b); }
    if (h_a != nullptr)
    { return generateOps_GEMM (self, h_a, index_a, B, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_dense <T> *A, const h_index *index_a, const dev_h_element <T> *B, const h_index *index_b) const
  {
    const dev_hierarchical <T> *h_b = B -> getElementHierarchical();
    const dev_low_rank <T> *lr_b = B -> getElementLowRank();
    const dev_dense <T> *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, d_b, index_b); }
    if (lr_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, lr_b, index_b); }
    if (h_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, h_b, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_low_rank <T> *A, const h_index *index_a, const dev_h_element <T> *B, const h_index *index_b) const
  {
    const dev_hierarchical <T> *h_b = B -> getElementHierarchical();
    const dev_low_rank <T> *lr_b = B -> getElementLowRank();
    const dev_dense <T> *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, d_b, index_b); }
    if (lr_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, lr_b, index_b); }
    if (h_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, h_b, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_hierarchical <T> *A, const h_index *index_a, const dev_h_element <T> *B, const h_index *index_b) const
  {
    const dev_hierarchical <T> *h_b = B -> getElementHierarchical();
    const dev_low_rank <T> *lr_b = B -> getElementLowRank();
    const dev_dense <T> *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, d_b, index_b); }
    if (lr_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, lr_b, index_b); }
    if (h_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, h_b, index_b); }

    return nullptr;
  }

  __host__ h_ops_tree * generateOps_GEMM (const h_index *self, const dev_h_element <T> *A, const h_index *index_a, const dev_h_element <T> *B, const h_index *index_b) const
  {
    const dev_hierarchical <T> *h_b = B -> getElementHierarchical();
    const dev_low_rank <T> *lr_b = B -> getElementLowRank();
    const dev_dense <T> *d_b = B -> getElementDense();

    if (d_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, d_b, index_b); }
    if (lr_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, lr_b, index_b); }
    if (h_b != nullptr)
    { return generateOps_GEMM (self, A, index_a, h_b, index_b); }

    return nullptr;
  }


   
  __host__ void print () const
  {
    printf("-- %d x %d | ld: %d --\n", ny, nx, ld);
    for (int y = 0; y < ny; y++)
    {
      for (int x = 0; x < nx; x++)
      {
        T e = elements[y * ld + x];
        printf("%.6f ", e);
      }
      printf("\n");
    }

    if (pivoted)
    {
      printf("\n-- Pivot: --\n");
      for (int y = 0; y < ny; y++)
      {
        printf("%d ", pivot[y]);
      }
      printf("\n");
    }
    
    printf("\n");
  }

  __host__ void loadTestMatrix(const int x_start = 0, const int y_start = 0)
  {
    for(int i = 0; i < ny; i++)
    {
      const int y = y_start + i;
      for(int j = 0; j < nx; j++)
      {
        const int x = x_start + j;
        elements[i * ld + j] = (T) (1.0 / (1.0 + ((x > y) ? x - y : y - x)));
      }
    }
  }

  __host__ void loadIdentityMatrix()
  {
    for (int x = 0; x < nx; x++) for (int y = 0; y < ny; y++)
    { elements[y * ld + x] = (T) ((x == y) ? 1 : 0); }
  }

  __host__ void loadRandomMatrix (const double min, const double max, const int seed = 0)
  {
    if (seed > 0) 
    { srand(seed); }
    for(int x = 0; x < nx; x++) for(int y = 0; y < ny; y++)
    { elements[y * ld + x] = (T) (min + ((T) rand() / RAND_MAX) * (max - min)); }
  }

  __host__ void loadRandomOrthMatrix (const int seed = 0)
  {
    if (seed > 0) 
    { srand(seed); }
    for (int x = 0; x < nx; x++) for (int y = 0; y < ny; y++)
    { elements[y * ld + x] = (T) ((x == y) ? 1 : 0); }

    for (int x = 0; x < nx; x++) for (int z = 0; z < x; z++)
    {
      double t = (double) rand() / RAND_MAX, cos = 1. / sqrt(1. + t * t), sin = cos * t;
      for (int y = 0; y < ny; y++)
      {
        const T e1_T = elements[y * ld + x], e2_T = elements[y * ld + z];
        elements[y * ld + x] = cos * e1_T - sin * e2_T;
        elements[y * ld + z] = sin * e1_T + cos * e2_T;
      }
    }
  }

  __host__ dev_dense <T> * matrixMultiplication (const dev_dense <T> *B) const
  {
    dev_dense <T> *C = new dev_dense <T> (B -> nx, ny);
    for(int m = 0; m < ny; m++)
    {
      for(int n = 0; n < B -> nx; n++)
      {
        for(int k = 0; k < nx; k++)
        {
          (C -> elements)[m * (C -> ld) + n] += elements[m * ld + k] * (B -> elements)[k * (B -> ld) + n];
        }
      }
    }
    return C;
  }

  __host__ dev_dense <T> * transpose() const
  {
    dev_dense <T> *C = new dev_dense <T> (ny, nx);
    for (int m = 0; m < ny; m++)
    {
      for (int n = 0; n < nx; n++)
      {
        (C -> elements)[n * (C -> ld) + m] = elements[m * ld + n];
      }
    }
    return C;
  }

  __host__ dev_dense <T> * restoreLU () const
  {
    dev_dense <T> *L = new dev_dense <T>(ny, ny);
    for (int i = 0; i < ny; i++)
    {
      for (int j = 0; j < ny; j++)
      {
        if (i > j && j < nx)
        {
          (L -> elements)[i * ny + j] = elements[i * ld + j];
        }
        else if (i == j)
        {
          (L -> elements)[i * ny + j] = 1;
        }
      }
    }

    dev_dense <T> *U = new dev_dense <T>(nx, ny);
    for (int i = 0; i < ny; i++)
    {
      for (int j = 0; j < nx; j++)
      {
        if (i <= j)
        {
          (U -> elements)[i * nx + j] = elements[i * ld + j];
        }
      }
    }
    
    dev_dense <T> *LU = L -> matrixMultiplication(U);
    delete L;
    delete U;
    return LU;
  }

  __host__ double sqrSum() const
  {
    double sum = 0.0;
    for (int x = 0; x < nx; x++)
    {
      for (int y = 0; y < ny; y++)
      {
        double t = (double)elements[y * ld + x];
        sum += t * t;
      }
    }
    return sum;
  }

  __host__ double L2Error (const dev_dense <T> *matrix) const
  {
    double norm = 0.0; int error_count = 0;
    for(int x = 0; x < nx; x++)
    {
      for(int y = 0; y < ny; y++)
      {
        double t = (double) (elements[y * ld + x] - (matrix -> elements)[y * (matrix -> ld) + x]);
        if (abs(t) > 1.e-6 && error_count < 10) 
        { printf("Error Location: (%d, %d). M1: %.6f M2: %.6f\n", y, x, elements[y * ld + x], (matrix -> elements)[y * (matrix -> ld) + x]); error_count++; }
        norm += t * t;
      }
    }
    return sqrt(norm / sqrSum());
  }

};


#endif