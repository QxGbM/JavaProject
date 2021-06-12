
cmake_minimum_required (VERSION 3.9)

project ("runtime" LANGUAGES CXX)

add_library(rts STATIC 
  dependency_checker.cpp dependency_checker.h
  dependency.cpp dependency.h
  ops_dag.cpp ops_dag.h
  ops.cpp ops.h
  scheduler.cpp scheduler.h)

target_include_directories(rts PRIVATE "./" )
