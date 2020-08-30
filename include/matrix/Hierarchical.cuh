
#pragma once
#ifndef _HIERARCHICAL
#define _HIERARCHICAL

#include <matrix/LowRank.cuh>

class Hierarchical : public Element {
private:
  vector<int> row_i;
  vector<int> col_i;
  vector<Element*> elements;

  void index_tree(const int m, const int n, const int part_y, const int part_x);

  bool in_tree(const int i, const int j) const;

public:
  
  Hierarchical(const int m, const int n, const int part_y, const int part_x);

  Hierarchical(const int m, const int n, const int part_y, const int part_x, const int abs_y, const int abs_x);

  ~Hierarchical();

  virtual Hierarchical* getElementHierarchical() override;

  virtual int getRowDimension() const;

  virtual int getColumnDimension() const;

  int getPartY() const;

  int getPartX() const;

  void setElement(Dense* d, const int i, const int j);

  void setElement(LowRank* lr, const int i, const int j);

  void setElement(Hierarchical* h, const int i, const int j);

  Element* getChild(const int i, const int j) const;

  void findChild(int& i, int& j, int& b_i, int& b_j) const;

  virtual Dense* convertToDense() const override;

  virtual void load(ifstream& stream) override;

  virtual void load(const real_t* arr, const int ld) override;

  virtual void print() const override;

  virtual void print(vector<int>& indices, vector<int>& config) const override;

};

#endif