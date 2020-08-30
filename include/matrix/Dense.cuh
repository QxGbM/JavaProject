
#pragma once
#ifndef _Dense
#define _Dense

#include <matrix/Element.cuh>

class Dense : public Element {
private:
  int m;
  int n;
  int ld;

  real_t* elements;

public:

  Dense(const int m, const int n, const int ld = 0);

  Dense(const int m, const int n, const int abs_y, const int abs_x, const int ld = 0);

  virtual ~Dense() override;

  virtual Dense* getElementDense() override;

  virtual int getRowDimension() const override;

  virtual int getColumnDimension() const override;

  virtual int getLeadingDimension() const override;

  virtual int getRank() const override;

  real_t* getElements() const;

  real_t* getElements(const int offset) const;

  real_t* getElements(real_t* host_ptr, const int ld) const;

  virtual void load(ifstream& stream) override;

  virtual void load(const real_t* arr, const int ld) override;

  virtual void print() const override;

  virtual void print(vector<int>& indices, vector<int>& config) const override;

  real_t sqrSum() const;

  real_t L2Error(const Dense& A) const;


};


#endif