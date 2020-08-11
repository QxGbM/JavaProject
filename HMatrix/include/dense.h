#pragma once
#ifndef _DEV_DENSE_CUH
#define _DEV_DENSE_CUH

#include <definitions.h>

class dense
{
private:
  int m;
  int n;
  int ld;

  real_t* elements;


public:

  dense(const int M, const int N);

  dense(const int M, const int N, const int LD);

  ~dense();

  int getRowDimension() const;

  int getColumnDimension() const;

  int getLeadingDimension() const;

  real_t* getElements(const int offset = 0) const;

  void resize(const int ld_in, const int ny_in);

  void resizeColumn(const int ld_in);

  void resizeRow(const int ny_in);

  void print() const;

  real_t sqrSum() const;

  real_t L2Error(const dense* matrix) const;

};


#endif