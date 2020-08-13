
#pragma once
#ifndef _HIERARCHICAL_H
#define _HIERARCHICAL_H

#include <definitions.h>
#include <element.h>

using std::vector;

class Hierarchical: public Element
{
private:

  int m;
  int n;
  vector<int> yIndices;
  vector<int> xIndices;
  vector<Element*> elements;

public:
  
  Hierarchical(const int M, const int N);

  ~Hierarchical();

  int getM() const;

  int getN() const;

  virtual int getRowDimension() const override;

  virtual int getColumnDimension() const override;

  void setElementDense(Dense* d, const int y, const int x);

  void setElementLowRank(LowRank* lr, const int y, const int x);

  void setElementHierarchical(Hierarchical* h, const int y, const int x);

  Element* getElement (const int y, const int x) const;

  virtual void print() const override;

  virtual void print(const int y, const int x, const int M, const int N) const override;


};

#endif