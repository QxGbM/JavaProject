
#include <definitions.h>
#include <dense.h>
#include <clusterbasis.h>

ClusterBasis::ClusterBasis(const int dim, const int rank) {
  basis = new Dense(dim, rank);
}

int ClusterBasis::getDimension() const {
  return basis->getRowDimension();
}

void ClusterBasis::print() const {
  print(0, 0, basis->getRowDimension(), basis->getColumnDimension());
}

void ClusterBasis::print(const int y, const int x, const int M, const int N) const {
  basis->print(y, x, M, N);
}