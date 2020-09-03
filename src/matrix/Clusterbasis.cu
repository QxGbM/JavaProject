
#include <matrix/clusterbasis.cuh>

Clusterbasis::Clusterbasis(const int dim, const int rank, const int* index, const int ld) {
  basis = new Dense(dim, rank, ld);
  Clusterbasis::index = index;
  child = list<Clusterbasis*>();
}

Clusterbasis::~Clusterbasis() {
  delete basis;
}

int Clusterbasis::getDimension() const {
  return basis->getRowDimension();
}

int Clusterbasis::getRank() const {
  return basis->getColumnDimension();
}

Dense* Clusterbasis::getBasis() {
  return basis;
}

real_t* Clusterbasis::getElements() {
  return basis->getElements();
}

void Clusterbasis::print() const {
  basis->print();
}
