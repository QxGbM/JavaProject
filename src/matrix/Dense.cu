
#include <matrix/Dense.cuh>
#include <algorithm>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime_api.h>


Dense::Dense (const int m, const int n, const int ld) : Element(element_t::dense, 0, 0) {
  using std::max;
  Dense::m = max(m, 0);
  Dense::n = max(n, 0);
  Dense::ld = max(m, ld);

  cudaMalloc(reinterpret_cast<void**>(&elements), (size_t)Dense::ld * Dense::n * sizeof(real_t));
  cudaMemset(elements, 0, (size_t)Dense::ld * Dense::n * sizeof(real_t));
}

Dense::Dense(const int m, const int n, const int abs_y, const int abs_x, const int ld) : Element(element_t::dense, abs_y, abs_x) {
  using std::max;
  Dense::m = max(m, 0);
  Dense::n = max(n, 0);
  Dense::ld = max(m, ld);

  cudaMalloc(reinterpret_cast<void**>(&elements), (size_t)Dense::ld * Dense::n * sizeof(real_t));
  cudaMemset(elements, 0, (size_t)Dense::ld * Dense::n * sizeof(real_t));
}

Dense::~Dense () {
  cudaFree (elements);
  Element::~Element();
}

Dense* Dense::getElementDense() {
  return this;
}

int Dense::getRowDimension () const
{ return m; }

int Dense::getColumnDimension () const
{ return n; }

int Dense::getLeadingDimension () const
{ return ld; }

int Dense::getRank() const {
  using std::min;
  return min(m, n);
}

real_t* Dense::getElements() const
{ return elements; }

real_t* Dense::getElements(const int offset) const
{ return &elements[offset]; }

real_t* Dense::getElements(real_t* host_ptr, const int ld) const {
  if (ld != m || Dense::ld != m) {
    cudaMemcpy2D(host_ptr, sizeof(real_t) * ld, elements, sizeof(real_t) * Dense::ld, sizeof(real_t) * m, n, cudaMemcpyDefault);
  }
  else {
    cudaMemcpy(host_ptr, elements, sizeof(real_t) * m * n, cudaMemcpyDefault);
  }
  return elements;
}

void Dense::load(ifstream& stream) {
  size_t real_l = sizeof(real_t);
  size_t n_e = real_l * m * n;
  unsigned char* buf = new unsigned char[n_e];
  stream.read(reinterpret_cast<char*>(buf), n_e);
  load(reinterpret_cast<real_t*>(buf), m);
  delete[] buf;
}

void Dense::load(const real_t* arr, const int ld) {
  if (ld != m || Dense::ld != m) {
    cudaMemcpy2D(elements, sizeof(real_t) * Dense::ld, arr, sizeof(real_t) * ld, sizeof(real_t) * m, n, cudaMemcpyDefault);
  }
  else {
    cudaMemcpy(elements, arr, sizeof(real_t) * m * n, cudaMemcpyDefault);
  }
}

void Dense::print() const {
  print(vector<int>(), vector<int> {0, getRowDimension(), 0, getColumnDimension()});
}
   
void Dense::print(vector <int>& indices, vector<int>& config) const {
  Element::print(indices);
  using std::cout;
  using std::endl;
  using std::max;
  using std::min;
  using std::fixed;
  cout << fixed;

  if (ld != m)
  { cout << m << " x " << n << " by " << ld << endl; }
  else
  { cout << m << " x " << n << endl; }

  int y_start = max(config[0], 0);
  int m = y_start + min(config[1], getRowDimension());
  int x_start = max(config[2], 0);
  int n = x_start + min(config[3], getColumnDimension());

  real_t* buf = new real_t[(size_t)m * n];
  getElements(buf, m);

  for (int y = y_start; y < m; y++) {
    for (int x = x_start; x < n; x++) {
      cout << buf[(size_t)x * m + y] << " ";
    }
    cout << endl;
  }
    
  cout << endl;
  delete[] buf;
}


real_t Dense::sqrSum() const {
  real_t* buf = new real_t[(size_t)m * n];
  getElements(buf, m);

  real_t sum = 0.0;
  for (int x = 0; x < n; x++) {
    for (int y = 0; y < m; y++) {
      real_t t = buf[(size_t)x * m + y];
      sum += t * t;
    }
  }
  delete[] buf;
  return sum;
}

real_t Dense::L2Error(const Dense& A) const {
  real_t* buf = new real_t[(size_t)m * n];
  real_t* buf_a = new real_t[(size_t)m * n];

  getElements(buf, m);
  A.getElements(buf_a, m);

  using std::cout;
  using std::endl;
  using std::abs;
  using std::sqrt;

  real_t norm = 0.;
  real_t sum = 0.;

  int error_count = 0;

  for(int x = 0; x < n; x++) {
    for(int y = 0; y < m; y++) {
      real_t t = buf[(size_t)x * m + y];
      real_t t_a = buf_a[(size_t)x * A.m + y];
      real_t diff = t - t_a;

      if (fabs(diff) > 1.e-8) {
        error_count++;
        if (error_count < 10) { 
          cout << "Error #" << error_count << ": (" << y  << ", " << x << "). M: " << t << " Ref: " << t_a << endl; 
        }
      }
      norm += diff * diff;
      sum += t_a * t_a;
    }
  }

  if (error_count > 0)
  { cout << "Total Error Locations: " << error_count << endl; }
  delete[] buf;
  delete[] buf_a;

  return sqrt(norm / sum);
}

/*

h_ops_tree * Dense::generateOps_GETRF (const h_index * self, dev_temp * tmp_mngr)
{
  h_ops_tree * op = new h_ops_tree (getrf, self);

  if (self -> getRank() > 0)
  { 
    op -> resizeChildren(2);
    h_index index_s = h_index (self); index_s.setShadow (self);
    h_ops_tree * op_accm = new h_ops_tree (accum, self, &index_s);
    op -> setChild(op_accm, 0);

    h_ops_tree * op_act = new h_ops_tree (getrf, self);
    op -> setChild(op_act, 1);

    delete op_accm;
    delete op_act;
  }

  return op;
}

h_ops_tree * Dense::generateOps_TRSML (const h_index * self, const Dense * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  h_ops_tree * op = new h_ops_tree (trsml, index_b, self);

  if (index_b -> getRank() > 0)
  { 
    op -> resizeChildren(2);
    h_index index_s = h_index (index_b); index_s.setShadow (index_b);
    h_ops_tree * op_accm = new h_ops_tree (accum, index_b, &index_s);
    op -> setChild(op_accm, 0);

    h_ops_tree * op_act = new h_ops_tree (trsml, index_b, self);
    op -> setChild(op_act, 1);

    delete op_accm;
    delete op_act;
  }

  return op;
}

h_ops_tree * Dense::generateOps_TRSML (const h_index * self, const LowRank * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  h_index index_lr = h_index (index_b); index_lr.setU();
  h_ops_tree * op = new h_ops_tree (trsml, &index_lr, self);
  return op;
}

h_ops_tree * Dense::generateOps_TRSML (const h_index * self, const Hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  printf("Not implemented.\n");
  return nullptr;
}

h_ops_tree * Dense::generateOps_TRSML (const h_index * self, const Element * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  const Hierarchical *h_b = B -> getElementHierarchical();
  const LowRank *lr_b = B -> getElementLowRank();
  const Dense *d_b = B -> getElementDense();

  if (d_b != nullptr)
  { return generateOps_TRSML (self, d_b, index_b, tmp_mngr); }
  if (lr_b != nullptr)
  { return generateOps_TRSML (self, lr_b, index_b, tmp_mngr); }
  if (h_b != nullptr)
  { return generateOps_TRSML (self, h_b, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * Dense::generateOps_TRSMR (const h_index * self, const Dense * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  h_ops_tree * op = new h_ops_tree (trsmr, index_b, self);

  if (index_b -> getRank() > 0)
  { 
    op -> resizeChildren(2);
    h_index index_s = h_index (index_b); index_s.setShadow (index_b);
    h_ops_tree * op_accm = new h_ops_tree (accum, index_b, &index_s);
    op -> setChild(op_accm, 0);

    h_ops_tree * op_act = new h_ops_tree (trsmr, index_b, self);
    op -> setChild(op_act, 1);

    delete op_accm;
    delete op_act;
  }

  return op;
}

h_ops_tree * Dense::generateOps_TRSMR (const h_index * self, const LowRank * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  h_index index_lr = h_index (index_b); index_lr.setVT();
  h_ops_tree * op = new h_ops_tree (trsmr, &index_lr, self);
  return op;
}

h_ops_tree * Dense::generateOps_TRSMR (const h_index * self, const Hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  printf("Not implemented.\n");
  return nullptr;
}

h_ops_tree * Dense::generateOps_TRSMR (const h_index * self, const Element * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  const Hierarchical *h_b = B -> getElementHierarchical();
  const LowRank *lr_b = B -> getElementLowRank();
  const Dense *d_b = B -> getElementDense();

  if (d_b != nullptr)
  { return generateOps_TRSMR (self, d_b, index_b, tmp_mngr); }
  if (lr_b != nullptr)
  { return generateOps_TRSMR (self, lr_b, index_b, tmp_mngr); }
  if (h_b != nullptr)
  { return generateOps_TRSMR (self, h_b, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * Dense::generateOps_ACCM (const h_index * self, const h_index * index_tmp_lr)
{
  if (self -> getRank() > 0)
  {
    h_index index_s = h_index (self); index_s.setShadow (self);
    return new h_ops_tree (accum, &index_s, index_tmp_lr);
  }
  else
  { return new h_ops_tree (accum, self, index_tmp_lr); }
}

h_ops_tree * Dense::generateOps_GEMM (const h_index * self, const Dense * A, const h_index * index_a, const Dense * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  return new h_ops_tree (gemm, self, index_a, index_b);
}

h_ops_tree * Dense::generateOps_GEMM (const h_index * self, const LowRank * A, const h_index * index_a, const Dense * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  if (self -> getRank() > 0)
  {
    h_index index_s = h_index (self); index_s.setShadow (self);
    return LowRank :: generateOps_GEMM (&index_s, A, index_a, B, index_b, tmp_mngr);
  }
  else
  { return new h_ops_tree (gemm, self, index_a, index_b); }
}

h_ops_tree * Dense::generateOps_GEMM (const h_index * self, const Hierarchical * A, const h_index * index_a, const Dense * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b);
  op -> resizeChildren(A -> getn_blocks() * A -> getm_blocks());

  int * y, * k, x = self -> getn(index_b -> getn());
  A -> getOffsets_y(&y);
  A -> getOffsets_x(&k);

#pragma omp parallel for if (omp_in_parallel() == 0)
  for (int i = 0; i < A -> getn_blocks() * A -> getm_blocks(); i++)
  {
    const int row = i / (A -> getn_blocks()), col = i - row * (A -> getn_blocks());
    const h_index index_ai = h_index (A, index_a, row, col), index_m = h_index (self, y[row], 0, index_ai.getm(), x), index_bj = h_index (index_b, k[col], 0, index_ai.getn(), x);
    h_ops_tree * op_i = generateOps_GEMM(&index_m, A -> getElement_blocks(row, col), &index_ai, B, &index_bj, tmp_mngr);
    op -> setChild(op_i, i);
    delete op_i;
  }

  delete[] y;
  delete[] k;
  return op;
}

h_ops_tree * Dense::generateOps_GEMM (const h_index * self, const Element * A, const h_index * index_a, const Dense * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  const Hierarchical *h_a = A -> getElementHierarchical();
  const LowRank *lr_a = A -> getElementLowRank();
  const Dense *d_a = A -> getElementDense();

  if (d_a != nullptr)
  { return generateOps_GEMM (self, d_a, index_a, B, index_b, tmp_mngr); }
  if (lr_a != nullptr)
  { return generateOps_GEMM (self, lr_a, index_a, B, index_b, tmp_mngr); }
  if (h_a != nullptr)
  { return generateOps_GEMM (self, h_a, index_a, B, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * Dense::generateOps_GEMM (const h_index * self, const Dense * A, const h_index * index_a, const LowRank * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  if (self -> getRank() > 0)
  {
    h_index index_s = h_index (self); index_s.setShadow (self);
    return LowRank :: generateOps_GEMM (&index_s, A, index_a, B, index_b, tmp_mngr);
  }
  else
  { return new h_ops_tree (gemm, self, index_a, index_b); }
}

h_ops_tree * Dense::generateOps_GEMM (const h_index * self, const LowRank * A, const h_index * index_a, const LowRank * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  if (self -> getRank() > 0)
  {
    h_index index_s = h_index (self); index_s.setShadow (self);
    return LowRank :: generateOps_GEMM (&index_s, A, index_a, B, index_b, tmp_mngr);
  }
  else
  { return new h_ops_tree (gemm, self, index_a, index_b); }
}

h_ops_tree * Dense::generateOps_GEMM (const h_index * self, const Hierarchical * A, const h_index * index_a, const LowRank * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  if (self -> getRank() > 0)
  {
    h_index index_s = h_index (self); index_s.setShadow (self);
    return LowRank :: generateOps_GEMM (&index_s, A, index_a, B, index_b, tmp_mngr);
  }
  else
  {
    h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b);
    op -> resizeChildren(A -> getn_blocks() * A -> getm_blocks());

    int * y, * k, x = self -> getn(index_b -> getn());
    A -> getOffsets_y(&y);
    A -> getOffsets_x(&k);

#pragma omp parallel for if (omp_in_parallel() == 0)
    for (int i = 0; i < A -> getn_blocks() * A -> getm_blocks(); i++)
    {
      const int row = i / (A -> getn_blocks()), col = i - row * (A -> getn_blocks());
      const h_index index_ai = h_index (A, index_a, row, col), index_m = h_index (self, y[row], 0, index_ai.getm(), x), index_bj = h_index (index_b, k[col], 0, index_ai.getn(), x);
      h_ops_tree * op_i = generateOps_GEMM(&index_m, A -> getElement_blocks(row, col), &index_ai, B, &index_bj, tmp_mngr);
      op -> setChild(op_i, i);
      delete op_i;
    }

    delete[] y;
    delete[] k;
    return op;
  }
}

h_ops_tree * Dense::generateOps_GEMM (const h_index * self, const Element * A, const h_index * index_a, const LowRank * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  const Hierarchical *h_a = A -> getElementHierarchical();
  const LowRank *lr_a = A -> getElementLowRank();
  const Dense *d_a = A -> getElementDense();

  if (d_a != nullptr)
  { return generateOps_GEMM (self, d_a, index_a, B, index_b, tmp_mngr); }
  if (lr_a != nullptr)
  { return generateOps_GEMM (self, lr_a, index_a, B, index_b, tmp_mngr); }
  if (h_a != nullptr)
  { return generateOps_GEMM (self, h_a, index_a, B, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * Dense::generateOps_GEMM (const h_index * self, const Dense * A, const h_index * index_a, const Hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b);
  op -> resizeChildren(B -> getn_blocks() * B -> getm_blocks());

  int * x, * k, y = self -> getm(index_a -> getm());
  B -> getOffsets_y(&k);
  B -> getOffsets_x(&x);

#pragma omp parallel for if (omp_in_parallel() == 0)
  for (int i = 0; i < B -> getn_blocks() * B -> getm_blocks(); i++)
  {
    const int row = i / (B -> getn_blocks()), col = i - row * (B -> getn_blocks());
    const h_index index_bj = h_index (B, index_b, row, col), index_m = h_index (self, 0, x[col], y, index_bj.getn()), index_ai = h_index (index_a, 0, k[row], y, index_bj.getm());
    h_ops_tree * op_i = generateOps_GEMM(&index_m, A, &index_ai, B -> getElement_blocks(row, col), &index_bj, tmp_mngr);
    op -> setChild(op_i, i);
    delete op_i;
  }

  delete[] x;
  delete[] k;
  return op;
}

h_ops_tree * Dense::generateOps_GEMM (const h_index * self, const LowRank * A, const h_index * index_a, const Hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  if (self -> getRank() > 0)
  {
    h_index index_s = h_index (self); index_s.setShadow (self);
    return LowRank :: generateOps_GEMM (&index_s, A, index_a, B, index_b, tmp_mngr);
  }
  else
  {
    h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b);
    op -> resizeChildren(B -> getn_blocks() * B -> getm_blocks());

    int * x, * k, y = self -> getm(index_a -> getm());
    B -> getOffsets_y(&k);
    B -> getOffsets_x(&x);

#pragma omp parallel for if (omp_in_parallel() == 0)
    for (int i = 0; i < B -> getn_blocks() * B -> getm_blocks(); i++)
    {
      const int row = i / (B -> getn_blocks()), col = i - row * (B -> getn_blocks());
      const h_index index_bj = h_index (B, index_b, row, col), index_m = h_index (self, 0, x[col], y, index_bj.getn()), index_ai = h_index (index_a, 0, k[row], y, index_bj.getm());
      h_ops_tree * op_i = generateOps_GEMM(&index_m, A, &index_ai, B -> getElement_blocks(row, col), &index_bj, tmp_mngr);
      op -> setChild(op_i, i);
      delete op_i;
    }

    delete[] x;
    delete[] k;
    return op;
  }
}

h_ops_tree * Dense::generateOps_GEMM (const h_index * self, const Hierarchical * A, const h_index * index_a, const Hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  const int n_k = A -> getn_blocks(); if (n_k != B -> getm_blocks())
  { printf("Matrices are partitioned differently in D.H-H GEMM.\n"); return nullptr; }

  h_ops_tree * op = new h_ops_tree (gemm, self, index_a, index_b);
  const int n_n = B -> getn_blocks(), n_mn = n_n * A -> getm_blocks(), n_mnk = n_mn * n_k;
  int * x, * y;
  A -> getOffsets_y(&y);
  B -> getOffsets_x(&x);

  op -> resizeChildren(n_mnk);

#pragma omp parallel for if (omp_in_parallel() == 0)
  for (int i = 0; i < n_mnk; i++)
  {
    const int k = i / n_mn, crd = i - k * n_mn, row = crd / n_n, col = crd - row * n_n;

    const h_index index_ai = h_index (A, index_a, row, k), index_bj = h_index (B, index_b, k, col);
    const h_index index_m = h_index (self, y[row], x[col], index_ai.getm(), index_bj.getn());
    h_ops_tree * op_k = generateOps_GEMM (&index_m, A -> getElement_blocks(row, k), &index_ai, B -> getElement_blocks(k, col), &index_bj, tmp_mngr);
    op -> setChild(op_k, i);
    delete op_k;
  }

  delete[] x;
  delete[] y;
  return op;
}

h_ops_tree * Dense::generateOps_GEMM (const h_index * self, const Element * A, const h_index * index_a, const Hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  const Hierarchical *h_a = A -> getElementHierarchical();
  const LowRank *lr_a = A -> getElementLowRank();
  const Dense *d_a = A -> getElementDense();

  if (d_a != nullptr)
  { return generateOps_GEMM (self, d_a, index_a, B, index_b, tmp_mngr); }
  if (lr_a != nullptr)
  { return generateOps_GEMM (self, lr_a, index_a, B, index_b, tmp_mngr); }
  if (h_a != nullptr)
  { return generateOps_GEMM (self, h_a, index_a, B, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * Dense::generateOps_GEMM (const h_index * self, const Dense * A, const h_index * index_a, const Element * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  const Hierarchical *h_b = B -> getElementHierarchical();
  const LowRank *lr_b = B -> getElementLowRank();
  const Dense *d_b = B -> getElementDense();

  if (d_b != nullptr)
  { return generateOps_GEMM (self, A, index_a, d_b, index_b, tmp_mngr); }
  if (lr_b != nullptr)
  { return generateOps_GEMM (self, A, index_a, lr_b, index_b, tmp_mngr); }
  if (h_b != nullptr)
  { return generateOps_GEMM (self, A, index_a, h_b, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * Dense::generateOps_GEMM (const h_index * self, const LowRank * A, const h_index * index_a, const Element * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  const Hierarchical *h_b = B -> getElementHierarchical();
  const LowRank *lr_b = B -> getElementLowRank();
  const Dense *d_b = B -> getElementDense();

  if (d_b != nullptr)
  { return generateOps_GEMM (self, A, index_a, d_b, index_b, tmp_mngr); }
  if (lr_b != nullptr)
  { return generateOps_GEMM (self, A, index_a, lr_b, index_b, tmp_mngr); }
  if (h_b != nullptr)
  { return generateOps_GEMM (self, A, index_a, h_b, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * Dense::generateOps_GEMM (const h_index * self, const Hierarchical * A, const h_index * index_a, const Element * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  const Hierarchical *h_b = B -> getElementHierarchical();
  const LowRank *lr_b = B -> getElementLowRank();
  const Dense *d_b = B -> getElementDense();

  if (d_b != nullptr)
  { return generateOps_GEMM (self, A, index_a, d_b, index_b, tmp_mngr); }
  if (lr_b != nullptr)
  { return generateOps_GEMM (self, A, index_a, lr_b, index_b, tmp_mngr); }
  if (h_b != nullptr)
  { return generateOps_GEMM (self, A, index_a, h_b, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * Dense::generateOps_GEMM (const h_index * self, const Element * A, const h_index * index_a, const Element * B, const h_index * index_b, dev_temp * tmp_mngr)
{
  const Hierarchical *h_b = B -> getElementHierarchical();
  const LowRank *lr_b = B -> getElementLowRank();
  const Dense *d_b = B -> getElementDense();

  if (d_b != nullptr)
  { return generateOps_GEMM (self, A, index_a, d_b, index_b, tmp_mngr); }
  if (lr_b != nullptr)
  { return generateOps_GEMM (self, A, index_a, lr_b, index_b, tmp_mngr); }
  if (h_b != nullptr)
  { return generateOps_GEMM (self, A, index_a, h_b, index_b, tmp_mngr); }

  return nullptr;
}

*/