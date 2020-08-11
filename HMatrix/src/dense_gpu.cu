
#include <cuda.h>
#include <cuda_runtime.h>
#include <definitions.h>
#include <dense.h>

real_t* Dense::copyToCudaArray(real_t* arr) const {
  real_t* e = arr;
  if (arr == nullptr) {
    cudaMalloc(&e, (size_t) m * n * sizeof(real_t));
  }

  if (ld == n) {
    cudaMemcpy(e, elements, (size_t)m * n * sizeof(real_t), cudaMemcpyHostToDevice);
  }
  else {
    for (int y = 0; y < m; y++) {
      cudaMemcpy(&e[y * n], &elements[y * ld], (size_t)n * sizeof(real_t), cudaMemcpyHostToDevice);
    }
  }
  return e;
}
