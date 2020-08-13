
#include <definitions.h>
#include <dense.h>

Dense::Dense(const int M, const int N) {
  m = M;
  n = N;
  ld = N;
  elements = vector<real_t>((size_t)M * N, 0);
}

Dense::Dense (const int M, const int N, const int LD) {
  m = M;
  n = N;
  ld = LD;
  elements = vector<real_t>((size_t)M * LD, 0);
}

Dense::~Dense () {}

int Dense::getRowDimension () const { 
  return m; 
}

int Dense::getColumnDimension () const { 
  return n; 
}

int Dense::getLeadingDimension () const { 
  return ld; 
}

real_t* Dense::getElements() {
  return elements.data();
}

real_t* Dense::getElements(const int offset) { 
  return &elements[offset]; 
}

void Dense::print() const {
  print(0, 0, m, n);
}
   
void Dense::print (const int y, const int x, const int M, const int N) const {
  using std::cout;
  using std::endl;
  using std::fixed;

  cout << "-- " << m << " x " << n << " | ld: " << ld << " | addr: " << elements.data() << " --" << endl << fixed;
  const int y_end_in = y + M, x_end_in = x + N;
  const int y_end = (y_end_in > m || y_end_in <= y) ? m : y_end_in, x_end = (x_end_in > n || x_end_in <= x) ? n : x_end_in;

  for (int i = y > 0 ? y : 0; i < y_end; i++)
  {
    for (int j = x > 0 ? x : 0; j < x_end; j++)
    {
      real_t e = elements[(size_t)i * ld + j];
      cout << e << " ";
    }
    cout << endl;
  }
    
  cout << endl;
}


real_t Dense::sqrSum() const {
  real_t sum = 0.0;
  for (int y = 0; y < m; y++) {
    for (int x = 0; x < n; x++) {
      real_t t = (real_t) elements[(size_t)y * ld + x];
      sum += t * t;
    }
  }
  return sum;
}

real_t Dense::L2Error(const Dense* matrix) const {
  using std::cout;
  using std::endl;
  using std::scientific;
  cout << scientific;

  real_t norm = 0.0; 
  int error_count = 0;
  for(int y = 0; y < m; y++) {
    for(int x = 0; x < n; x++) {
      real_t val1 = elements[(size_t)y * ld + x];
      real_t val2 = (matrix->elements)[(size_t)y * (matrix->ld) + x];
      real_t t = val1 - val2;
      if (fabs(t) > 1.e-8) {
        if (error_count < 10) { 
          cout << "Error Location: (" << y << ", " << x << "). M1: " << val1 << "M2: " << val2 << endl; 
        }
        error_count ++;
      }
      norm += t * t;
    }
  }

  if (error_count > 0) { 
    cout << "Total Error Locations: " << error_count << endl; 
  }
  return sqrt(norm / sqrSum());
}


Dense* Dense::getElementDense() {
  return this;
}