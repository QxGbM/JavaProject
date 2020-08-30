
#pragma once
#ifndef _INST
#define _INST

#include <matrix/Element.cuh>

enum class op_t {
  getrf = 0,
  trsml = 1,
  trsmr = 2,
  gemm = 3,
  accm = 4,
  nop = -1
};

class instruction {
protected:
  op_t op;
  vector<int> params;
  vector<real_t*> dataptrs;
  vector<int> locs;
  vector<element_t> types;

public:
  instruction() {
    op = op_t::nop;
  }

  ~instruction() {
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