
#include <cuda.h>
#include <cuda_runtime.h>
#include <definitions.h>
#include <dense.h>

real_t* Dense::copyToCudaArray() const {
  return copyToCudaArray(nullptr, ld);
}

real_t* Dense::copyToCudaArray(real_t* arr, const int ld_arr) const {
  real_t* e = arr;
  if (arr == nullptr) {
    cudaMalloc(&e, (size_t)m * ld_arr * sizeof(real_t));
  }

  if (ld == ld_arr) {
    cudaMemcpy(e, elements.data(), (size_t)m * ld * sizeof(real_t), cudaMemcpyHostToDevice);
  }
  else {
    for (int y = 0; y < m; y++) {
      cudaMemcpy(&e[(size_t)y * ld_arr], &elements.data()[(size_t)y * ld], (size_t)n * sizeof(real_t), cudaMemcpyHostToDevice);
    }
  }
  return e;
}

void Dense::copyFromCudaArray(real_t* arr, const int ld_arr) {
  if (ld == ld_arr) {
    cudaMemcpy(elements.data(), arr, (size_t)m * ld * sizeof(real_t), cudaMemcpyDeviceToHost);
  }
  else {
    for (int y = 0; y < m; y++) {
      cudaMemcpy(&elements.data()[(size_t)y * ld], &arr[(size_t)y * ld_arr], (size_t)n * sizeof(real_t), cudaMemcpyDeviceToHost);
    }
  }

}