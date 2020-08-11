
#include <definitions.h>
#include <dense.h>

Dense::Dense(const int M, const int N) {
  Dense(M, N, N);
}

Dense::Dense (const int M, const int N, const int LD) {
  m = M;
  n = N;
  ld = LD;
  elements = new real_t[(size_t) M * LD];
  memset(elements, 0, (size_t) M * LD);
}

Dense::~Dense () {
  delete[] elements;
}

int Dense::getRowDimension () const { 
  return m; 
}

int Dense::getColumnDimension () const { 
  return n; 
}

int Dense::getLeadingDimension () const { 
  return ld; 
}

real_t* Dense::getElements (const int offset) const { 
  return &elements[offset]; 
}

void Dense::resize (const int LD, const int M) {
  resizeColumn(LD);
  resizeRow(M);
}

void Dense::resizeColumn (const int LD) {
  if (LD > 0 && LD != ld) {
    real_t* e = new real_t[(size_t) m * LD];

    for (int y = 0; y < m; y++) {
      for (int x = 0; x < n && x < LD; x++) {
        e[y * LD + x] = elements[y * ld + x];
      }
    }

    delete[] elements;
    ld = LD;
    n = n > ld ? ld : n;
    elements = e;
  }
}

void Dense::resizeRow (const int M) {
  if (M > 0 && M != m) {
    real_t * e = new real_t[(size_t) M * ld];

    for (int y = 0; y < M && y < m; y++) {
      for (int x = 0; x < n; x++) {
        e[y * ld + x] = elements[y * ld + x];
      }
    }

    delete[] elements;
    elements = e;
    m = M;
  }
}

void Dense::print() const {
  print(0, 0, m, n);
}
   
void Dense::print (const int y, const int x, const int M, const int N) const {
  printf("-- %d x %d | ld: %d | addr: %p --\n", m, n, ld, elements);
  const int y_end_in = y + M, x_end_in = x + N;
  const int y_end = (y_end_in > m || y_end_in <= y) ? m : y_end_in, x_end = (x_end_in > n || x_end_in <= x) ? n : x_end_in;

  for (int i = y > 0 ? y : 0; i < y_end; i++)
  {
    for (int j = x > 0 ? x : 0; j < x_end; j++)
    {
      real_t e = elements[i * ld + j];
      printf("%.6e ", e);
    }
    printf("\n");
  }
    
  printf("\n");
}


real_t Dense::sqrSum() const {
  real_t sum = 0.0;
  for (int y = 0; y < m; y++) {
    for (int x = 0; x < n; x++) {
      real_t t = (real_t) elements[y * ld + x];
      sum += t * t;
    }
  }
  return sum;
}

real_t Dense::L2Error (const Dense* matrix) const
{
  real_t norm = 0.0; 
  int error_count = 0;
  for(int y = 0; y < m; y++) {
    for(int x = 0; x < n; x++) {
      real_t val1 = elements[y * ld + x];
      real_t val2 = (matrix->elements)[y * (matrix->ld) + x];
      real_t t = (real_t) (val1 - val2);
      if (fabs(t) > 1.e-8) {
        if (error_count < 10)
        { printf("Error Location: (%d, %d). M1: %.6e M2: %.6e\n", y, x, val1, val2); }
        error_count ++;
      }
      norm += t * t;
    }
  }

  if (error_count > 0)
  { printf("Total Error Locations: %d.\n", error_count); }
  return sqrt(norm / sqrSum());
}


real_t* Dense::copyToArray(real_t* arr) const {
  real_t* e = arr == nullptr ? new real_t[(size_t)m * n] : arr;
  for (int y = 0; y < m; y++) {
    for (int x = 0; x < n; x++) {
      e[y * n + x] = elements[y * ld + x];
    }
  }
  return e;
}

