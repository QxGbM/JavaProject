

#include <Dense.h>
#include <algorithm>
#include <cstdlib>
#include <cassert>

Dense::Dense (int64_t m_, int64_t n_, int64_t lda_) : m(m_), n(n_), lda(std::max(m, lda_)) {

  a = (double*)malloc(sizeof(double) * lda * n);
}

Dense::~Dense () {
  if (a)
    free(a);
}