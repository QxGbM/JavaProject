
#pragma once
#ifndef _Element_CUH
#define _Element_CUH

#include <iostream>
#include <fstream>
#include <vector>

#define real_t double

enum class element_t {
  empty = 0,
  dense = 1,
  low_rank = 2,
  hierarchical = 3,
  accumulator = 4
};

class Dense;
class LowRank;
class Hierarchical;

using std::ifstream;
using std::vector;

class Element {
protected:
  element_t type;

  int abs_y;
  int abs_x;

  Dense* accum_u;
  Dense* accum_v;

public:
  
  Element(const element_t type, const int abs_y, const int abs_x);

  virtual ~Element();

  virtual Dense* getElementDense() const;

  virtual LowRank* getElementLowRank() const;

  virtual Hierarchical* getElementHierarchical() const;

  element_t getType() const;

  virtual int getRowDimension() const;

  virtual int getColumnDimension() const;

  virtual int getLeadingDimension() const;

  virtual int getRank() const;

  virtual real_t getElement (const int i, const int j) const;

  virtual void setAccumulator(const int rank);

  void setAccumulator(Dense& U, Dense& V);

  Dense* getAccumulatorU();

  Dense* getAccumulatorV();

  virtual Dense* convertToDense() const;

  virtual void loadBinary (ifstream& stream);

  virtual void print(vector<int>& indices, vector<int>& config) const;

  void print(vector<int>& indices) const;

};


#endif