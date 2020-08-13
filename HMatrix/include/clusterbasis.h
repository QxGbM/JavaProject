
#pragma once
#ifndef _CLUSTERBASIS_H
#define _CLUSTERBASIS_H

#include <definitions.h>
#include <dense.h>

class ClusterBasis
{
private:
  Dense* basis;

public:

  ClusterBasis (const int dim, const int rank);

  int getDimension() const;

  void print() const;

  void print(const int y, const int x, const int M, const int N) const;

};


#endif
