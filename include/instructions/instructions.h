
#pragma once
#ifndef _Inst
#define _Inst

#include <vector>

enum class matrix_t;

enum class op_t {
  getrf = 0,
  trsml = 1,
  trsmr = 2,
  gemm = 3,
  accm = 4
};

using std::vector;

class instruction {
protected:
  op_t op;
  vector<int> params;
  vector<void*> dataptrs;
  vector<int> locs;
  vector<matrix_t> types;

public:
  virtual ~instruction() {
  }
  
  virtual void getReadWriteRange(int& y, int& x, int& m, int& n) const {
    y = 0;
    x = 0;
    m = 0;
    n = 0;
  }

  virtual void getReadOnlyRange(int i, int& y, int& x, int& m, int& n) const {
    y = 0;
    x = 0;
    m = 0;
    n = 0;
  }

  

};

#endif