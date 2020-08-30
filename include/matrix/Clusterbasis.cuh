
#pragma once
#ifndef _Clusterbasis
#define _Clusterbasis

#include <matrix/Dense.cuh>
#include <list>

using std::list;

class Clusterbasis {
private:
  Dense* basis;
  const int* index;
  list<Clusterbasis*> child;

public:

  Clusterbasis(const int dim, const int rank, const int* index, const int ld = 0);

  ~Clusterbasis();

  int getDimension() const;

  int getRank() const;

  void print() const;

};


#endif
