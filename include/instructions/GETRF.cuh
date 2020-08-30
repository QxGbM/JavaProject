
#pragma once
#ifndef _GETRF
#define _GETRF

#include <instructions/instructions.cuh>
#include <matrix/Dense.cuh>

class GETRF : public instruction {
public:
  GETRF(Dense& d) {
    op = op_t::getrf;
    params = vector<int>{ d.getRowDimension(), d.getColumnDimension(), d.getLeadingDimension() };
    dataptrs = vector<real_t*>{ d.getElements() };
    int y, x;
    d.getLocs(y, x);
    locs = vector<int>{ y, x };
    types = vector<element_t>{ element_t::dense };
  }

  ~GETRF() {
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