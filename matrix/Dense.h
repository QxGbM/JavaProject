
#pragma once

#include <cstdint>

class Dense {
public:
  int64_t m;
  int64_t n;
  int64_t lda;

  double* a;

  Dense(int64_t m, int64_t n, int64_t lda);

  ~Dense();

};
