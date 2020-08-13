
#pragma once
#ifndef _DENSE_H
#define _DENSE_H

#include <definitions.h>
#include <element.h>

class Dense: public Element
{
private:
  int m;
  int n;
  int ld;

  real_t * elements;

public:
  
  Dense(const int M, const int N);

  Dense(const int M, const int N, const int LD);

  ~Dense();

  virtual int getRowDimension() const override;

  virtual int getColumnDimension() const override;

  int getLeadingDimension() const;

  real_t* getElements(const int offset = 0) const;

  void resize(const int LD, const int M);

  void resizeColumn(const int LD);

  void resizeRow(const int M);

  virtual void print() const override;

  virtual void print(const int y, const int x, const int M, const int N) const override;

  real_t sqrSum() const;

  real_t L2Error(const Dense* matrix) const;

  real_t* copyToArray(real_t* arr = nullptr) const;

  real_t* copyToCudaArray(real_t* arr = nullptr) const;

  virtual Dense* getElementDense() override;

};


#endif