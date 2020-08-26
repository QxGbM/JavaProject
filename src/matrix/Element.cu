
#include <matrix/Element.cuh>
#include <matrix/Dense.cuh>

Element::Element(const element_t type, const int abs_y, const int abs_x) {
  Element::type = type;
  Element::abs_y = abs_y;
  Element::abs_x = abs_x;
  accum_u = nullptr;
  accum_v = nullptr;
}

Element::~Element() { 
  if (accum_u != nullptr)
  { delete accum_u; }
  if (accum_v != nullptr)
  { delete accum_v; }
}

Dense* Element::getElementDense() const {
  return nullptr;
}

LowRank* Element::getElementLowRank() const {
  return nullptr;
}

Hierarchical* Element::getElementHierarchical() const {
  return nullptr;
}

element_t Element::getType() const { 
  return type; 
}

int Element::getRowDimension() const {
  return 0;
}

int Element::getColumnDimension() const {
  return 0;
}

int Element::getLeadingDimension() const {
  return 0;
}

int Element::getRank() const {
  return 0;
}

real_t Element::getElement (const int i, const int j) const {
  return 0.;
}

void Element::setAccumulator(const int rank) {

}

void Element::setAccumulator(Dense& U, Dense& V) {
  accum_u = &U;
  accum_v = &V;
}

Dense* Element::getAccumulatorU() {
  return accum_u;
}

Dense* Element::getAccumulatorV() {
  return accum_v;
}

Dense* Element::convertToDense() const {

}

void Element::loadBinary (ifstream& stream) {
  
}

void print(vector<int>& indices, vector<int>& config) {

}

void print(vector<int>& indices) {
  using std::cout;
  using std::endl;
  for (auto iter = indices.begin(); iter != indices.end(); iter++)
  { cout << *iter; }
  cout << endl;
}