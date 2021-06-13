
#pragma once

#include <vector>
#include <cstddef>

enum class dependency_t;

class DataMap2D {
public:
  void* ptr;
  size_t pitch;
  size_t width;
  size_t height;

  DataMap2D(void* ptr, size_t pitch, size_t width, size_t height) : ptr(ptr), pitch(pitch), width(width), height(height) {}

  bool checkOverlap(const DataMap2D& data) const;

};

class Operation {
private:
  std::vector<std::pair<bool, DataMap2D>> data;

public:

  Operation() : data() {}

  ~Operation() {}

  dependency_t checkDependencyFrom(const Operation& op_from) const;

  dependency_t checkDependencyTo(const Operation& op_to) const;
  
};

