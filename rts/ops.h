
#pragma once

#include <vector>

enum class dependency_t;

class DataMap2D {
public:
  void* ptr;
  size_t pitch;
  size_t width;
  size_t height;

  DataMap2D(void* ptr, size_t pitch, size_t width, size_t height) : ptr(ptr), pitch(pitch), width(width), height(height) {}
};

class Operation {
public:
  std::vector<bool> ro_rw;
  std::vector<DataMap2D> data;

  Operation() : ro_rw(), data() {}

  ~Operation() {}

  dependency_t checkDependencyFrom(const Operation& op_from) const;

  dependency_t checkDependencyTo(const Operation& op_to) const;
  
};

