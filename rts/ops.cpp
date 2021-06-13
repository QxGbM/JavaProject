

#include <ops.h>
#include <dependency.h>
#include <algorithm>

bool DataMap2D::checkOverlap(const DataMap2D& data) const {

  unsigned char* ptr_a = reinterpret_cast<unsigned char*>(ptr);
  unsigned char* ptr_b = reinterpret_cast<unsigned char*>(data.ptr);

  auto diff = ptr_b - ptr_a;
  if (diff < 0)
    return data.checkOverlap(*this);
  else if ((size_t)diff >= height * pitch)
    return false;
  else {
    bool ol = ptr_a == ptr_b;
    unsigned char* cmp_fin = ptr_b + data.height * data.pitch;
    size_t i_end = std::min(height, ((size_t)(cmp_fin - ptr_a) + pitch - 1) / pitch);

    for (size_t i = diff / pitch; i < i_end && !ol; i++) {
      auto data_start = ptr_a + i * pitch;
      auto data_end = data_start + width;

      auto diff_b = data_start - ptr_b;
      auto cmp_start = diff_b > 0 ? ptr_b + (diff_b / data.pitch) * data.pitch : ptr_b;

      while (cmp_start < data_end && cmp_start < cmp_fin && !ol) {
        auto cmp_end = cmp_start + data.width;
        if (!(cmp_end <= data_start || data_end <= cmp_start))
          ol = true;
        cmp_start = cmp_start + data.pitch;
      }

    }

    return ol;
  }
}

dependency_t Operation::checkDependencyFrom (const Operation& op_from) const {
  dependency_t dep = dependency_t::no;

  for (size_t i = 0; i < op_from.data.size(); i++)
    for (size_t j = 0; j < data.size(); j++) {

      bool ol = op_from.data[i].second.checkOverlap(data[j].second);

      if (ol) {
        if (!op_from.data[i].first && data[j].first)
          dep = dep + dependency_t::flow;
        if (op_from.data[i].first && !data[j].first)
          dep = dep + dependency_t::anti;
        if (!op_from.data[i].first && !data[j].first)
          dep = dep + dependency_t::out;
      }
    }

  return dep;
}

dependency_t Operation::checkDependencyTo (const Operation& op_to) const
{ return op_to.checkDependencyFrom(*this); }
