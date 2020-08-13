
#include <definitions.h>
#include <dense.h>
#include <lowrank.h>
#include <hierarchical.h>

using std::vector;

Hierarchical::Hierarchical(const int M, const int N) {
  m = M;
  n = N;
  yIndices = vector<int>(m, 0);
  xIndices = vector<int>(n, 0);
  elements = vector<Element*>((size_t) m * n, nullptr);
}

Hierarchical::~Hierarchical() {
  for(int i = 0; i < m * n; i++)
  { delete elements[i]; }
}

int Hierarchical::getM() const {
  return m;
}

int Hierarchical::getN() const {
  return n;
}

int Hierarchical::getRowDimension() const {
  int dim = 0;
  for (int i = 0; i < m; i++) {
    dim += elements[(size_t)i * n]->getRowDimension();
  }
  return dim;
}

int Hierarchical::getColumnDimension() const {
  int dim = 0;
  for (int i = 0; i < n; i++) {
    dim += elements[i]->getColumnDimension();
  }
  return dim;
}

void Hierarchical::setElementDense(Dense* d, const int y, const int x) {
  elements[(size_t)y * n + x] = dynamic_cast<Element*>(d);
}

void Hierarchical::setElementLowRank(LowRank* lr, const int y, const int x) {
  elements[(size_t)y * n + x] = dynamic_cast<Element*>(lr);
}

void Hierarchical::setElementHierarchical(Hierarchical* h, const int y, const int x) {
  elements[(size_t)y * n + x] = dynamic_cast<Element*>(h);
}

Element* Hierarchical::getElement(const int y, const int x) const {
  return dynamic_cast<Element*>(elements[(size_t)y * n + x]);
}

void Hierarchical::print() const {
  print(0, 0, getRowDimension(), getColumnDimension());
}

void Hierarchical::print(const int y, const int x, const int M, const int N) const {
  using std::cout;
  for (int i = 0; i < m * n; i++) {
    elements[i]->print(y, x, M, N);
  }
}
