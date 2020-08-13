
#include <definitions.h>
#include <dense.h>
#include <clusterbasis.h>
#include <lowrank.h>

LowRank::LowRank(const int M, const int N, const int rank) {
  U = new ClusterBasis(M, rank);
  VT = new ClusterBasis(N, rank);
  S = new Dense(rank, rank);
}

LowRank::LowRank(const Dense* d) {

}

LowRank::~LowRank() {
  delete U;
  delete VT;
  delete S;
}

int LowRank::getRowDimension() const {
  return U->getDimension();
}

int LowRank::getColumnDimension() const {
  return VT->getDimension();
}

int LowRank::getRank() const {
  return S->getRowDimension();
}

ClusterBasis* LowRank::getU() const {
  return U;
}

ClusterBasis* LowRank::getVT() const {
  return VT;
}

Dense* LowRank::getS() const {
  return S;
}

void LowRank::print() const {
  print(0, 0, getRowDimension(), getColumnDimension());
}

void LowRank::print(const int y, const int x, const int M, const int N) const {
  int rank = getRank();
  U->print(y, 0, M, rank);
  VT->print(x, 0, N, rank);
  S->print(0, 0, rank, rank);
}

LowRank* LowRank::getElementLowRank() {
  return this;
}