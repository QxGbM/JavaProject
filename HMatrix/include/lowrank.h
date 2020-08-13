
#pragma once
#ifndef _LOWRANK_H
#define _LOWRANK_H

#include <definitions.h>
#include <dense.h>
#include <clusterbasis.h>

class LowRank: public Element
{
private:
  ClusterBasis* U;
  ClusterBasis* VT;
  Dense* S;

public:

  LowRank (const int M, const int N, const int rank);

  LowRank (const Dense* d);

  ~LowRank ();

  virtual int getRowDimension() const override;

  virtual int getColumnDimension() const override;

  int getRank () const;

  ClusterBasis* getU () const;

  ClusterBasis* getVT () const;

  Dense* getS() const;

  virtual void print() const override;

  virtual void print(const int y, const int x, const int M, const int N) const override;

  virtual LowRank* getElementLowRank() override;
};


#endif
