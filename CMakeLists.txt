
cmake_minimum_required (VERSION 3.9)

project ("shimo" LANGUAGES CXX CUDA)

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(./matrix ./rts)

add_subdirectory(./matrix)
add_subdirectory(./rts)

add_executable(main main.cpp)
target_link_libraries(main rts)
set_property(TARGET main PROPERTY CUDA_ARCHITECTURES 60 70 75)
