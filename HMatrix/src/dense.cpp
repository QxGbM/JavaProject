
#include <definitions.h>
#include <dense.h>

dense::dense(const int M, const int N) {
  m = M;
  n = N;
  ld = N;
  elements = new real_t[(size_t) M * N];
  memset(elements, 0, (size_t) M * N * sizeof(real_t));
}

dense::dense(const int M, const int N, const int LD) {
  m = M;
  n = N;
  ld = LD;
  elements = new real_t[(size_t) M * LD];
  memset(elements, 0, (size_t) M * LD * sizeof(real_t));
}

dense::~dense() {
  delete[] elements;
}

int dense::getRowDimension() const {
  return m;
}

int dense::getColumnDimension() const {
  return n;
}

int dense::getLeadingDimension() const {
  return ld;
}

real_t* dense::getElements(const int offset) const {
  return &elements[offset];
}

void resize(const int ld_in, const int ny_in);

void resizeColumn(const int ld_in);

void resizeRow(const int ny_in);

void print() const;

real_t sqrSum() const;

real_t L2Error(const dense* matrix) const;