
#pragma once
#ifndef _LOWRANK_H
#define _LOWRANK_H

#include <definitions.h>
#include <dense.h>
#include <clusterbasis.h>

class LowRank 
{
private:
  ClusterBasis* U;
  ClusterBasis* VT;
  Dense* S;

public:

  LowRank (const int M, const int N, const int rank);

  LowRank (const Dense* d);

  ~LowRank ();

  int getRowDimension () const;

  int getColumnDimension () const;

  int getRank () const;

  ClusterBasis* getU () const;

  ClusterBasis* getVT () const;

  Dense* getS() const;

  void print() const;

  void print(const int y, const int x, const int M, const int N) const;
};


#endif
