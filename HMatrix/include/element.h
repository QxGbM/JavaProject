
#pragma once
#ifndef _ELEMENT_H
#define _ELEMENT_H

#include <definitions.h>

class Dense;
class LowRank;

class Element 
{
public:
  
  virtual ~Element() {}

  virtual int getRowDimension() const {
    return 0;
  }

  virtual int getColumnDimension() const {
    return 0;
  }

  virtual Dense* getElementDense() {
    return nullptr;
  }

  virtual LowRank* getElementLowRank() {
    return nullptr;
  }

  virtual void print() const {

  }

  virtual void print(const int y, const int x, const int M, const int N) const {

  }


};


#endif