
#pragma once
#ifndef _Element
#define _Element

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

  virtual Dense* getElementDense();

  virtual LowRank* getElementLowRank();

  virtual Hierarchical* getElementHierarchical();

  virtual int getRowDimension() const;

  virtual int getColumnDimension() const;

  virtual int getLeadingDimension() const;

  virtual int getRank() const;

  virtual Dense* convertToDense() const;

  virtual void load(ifstream& stream);

  virtual void load(const real_t* arr, const int ld);

  virtual void print() const;

  virtual void print(vector<int>& indices, vector<int>& config) const;

  element_t getType() const;

  void getLocs(int& abs_y, int& abs_x) const;

  void setLocs(const int abs_y, const int abs_x);

  bool admissible(const double condition) const;

  void setAccumulator(const int rank);

  void setAccumulator(Dense& U, Dense& V);

  Dense* getAccumulatorU();

  Dense* getAccumulatorV();

  void print(vector<int>& indices) const;

};


#endif