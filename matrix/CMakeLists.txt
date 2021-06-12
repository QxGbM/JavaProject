
cmake_minimum_required (VERSION 3.9)

project ("matrix" LANGUAGES CXX CUDA)

add_library(matrix STATIC 
  Dense.cpp Dense.h 
  Hierarchical.cpp Hierarchical.h )

target_include_directories(matrix PRIVATE "./")