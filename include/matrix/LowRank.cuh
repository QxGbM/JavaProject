
#pragma once
#ifndef _LowRank
#define _LowRank

#include <matrix/Clusterbasis.cuh>

class LowRank : public Element {
private:
  Clusterbasis* U;
  Clusterbasis* V;
  Dense* S;

public:

  LowRank (const int m, const int n, const int rank, const int ld = 0);

  LowRank(const int m, const int n, const int rank, const int abs_y, const int abs_x, const int ld = 0);

  ~LowRank ();

  virtual LowRank* getElementLowRank() override;

  virtual int getRowDimension() const override;

  virtual int getColumnDimension() const override;

  virtual int getRank() const override;

  virtual Dense* convertToDense() const override;

  virtual void print() const override;

  virtual void print(vector<int>& indices, vector<int>& config) const override;

};


#endif
