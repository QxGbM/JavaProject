
cmake_minimum_required (VERSION 3.9)

project ("runtime" LANGUAGES CXX)

add_library(rts STATIC 
  dependency.cpp dependency.h
  ops.cpp ops.h
  scheduler.cpp scheduler.h
  )

target_include_directories(rts PRIVATE "./" )
