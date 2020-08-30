
#include <matrix/Element.cuh>
#include <matrix/Dense.cuh>

Element::Element(const element_t type, const int abs_y, const int abs_x) {
  Element::type = type;
  setLocs(abs_y, abs_x);
  accum_u = nullptr;
  accum_v = nullptr;
}

Element::~Element() { 
  if (accum_u != nullptr)
  { delete accum_u; }
  if (accum_v != nullptr)
  { delete accum_v; }
}

Dense* Element::getElementDense() {
  return nullptr;
}

LowRank* Element::getElementLowRank() {
  return nullptr;
}

Hierarchical* Element::getElementHierarchical() {
  return nullptr;
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

Dense* Element::convertToDense() const {
  return nullptr;
}

void Element::load(ifstream& stream) {
  
}

void Element::load(const real_t* arr, const int ld) {

}

void Element::print() const {

}

void Element::print(vector<int>& indices, vector<int>& config) const {

}

element_t Element::getType() const {
  return type;
}

void Element::getLocs(int& abs_y, int& abs_x) const {
  abs_y = Element::abs_y;
  abs_x = Element::abs_x;
}

void Element::setLocs(const int abs_y, const int abs_x) {
  Element::abs_y = abs_y;
  Element::abs_x = abs_x;
}

bool Element::admissible(const double condition) const {
  using std::abs;
  using std::min;
  return condition * abs(abs_y - abs_x) > min(getRowDimension(), getColumnDimension());
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

void Element::print(vector<int>& indices) const {
  using std::cout;
  using std::endl;
  for (auto iter = indices.begin(); iter != indices.end(); iter++)
  { cout << *iter; }
  cout << endl;
}