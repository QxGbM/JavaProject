
#pragma once
#ifndef _Getrf
#define _Getrf

#include <instructions/instructions.h>
#include <matrix/Dense.cuh>

enum class matrix_t;
using std::vector;

class getrf : protected instruction {
public:
  getrf(Dense& d, int y, int x) {
    op = op_t::getrf;
    params = vector<int>{ d.getNy(), d.getNx(), d.getLd() };
    dataptrs = vector<void*>{ reinterpret_cast<void*>(d.getElements()) };
    locs = vector<int>{ y, x };
    types = vector<matrix_t>{ matrix_t::dense };
  }

  virtual ~getrf() override {
  }
  
  virtual void getReadWriteRange (int& y, int& x, int& m, int& n) const override {
    y = locs[0];
    x = locs[1];
    m = params[0];
    n = params[1];
  }

  virtual void getReadOnlyRange (int i, int& y, int& x, int& m, int& n) const override {
    y = locs[0];
    x = locs[1];
    m = params[0];
    n = params[1];
  }

  

};

#endif