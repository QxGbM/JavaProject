
#include <matrix/Compressor.cuh>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <omp.h>

const int n_stream = 4;

compressor::compressor(Hierarchical& h, const int rank, const double condition) {
  load(h, rank, condition);
  compress();
}

compressor::~compressor() {
  for (auto iter = d_lis.begin(); iter != d_lis.end(); iter++) {
    delete* iter;
  }
}

void compressor::load(Hierarchical& h, const int rank, const double condition) {
  for (int x = 0; x < h.getPartX(); x++) {
    for (int y = 0; y < h.getPartY(); y++) {
      Element* e = h.getChild(y, x);
      if (e != nullptr && e->getElementHierarchical() != nullptr) {
        load(*(e->getElementHierarchical()), rank, condition);
      }
      else if (e != nullptr && e->getElementDense() != nullptr) {
        bool admis = e->admissible(condition);
        if (admis) {
          d_lis.push_back(e->getElementDense());
          LowRank* lr = new LowRank(e->getRowDimension(), e->getColumnDimension(), rank);
          lr_lis.push_back(lr);
          h.setElement(lr, y, x);
        }
      }
    }
  }
}

void compressor::compress() {
  cudaStream_t streams[n_stream];
  cusolverDnHandle_t shandle[n_stream];
  cublasHandle_t chandle[n_stream];
  real_t* workspace[n_stream];

  for (int i = 0; i < n_stream; i++) {
    cudaStreamCreate(&streams[i]);
    cusolverDnCreate(&shandle[i]);
    cublasCreate(&chandle[i]);
    cusolverDnSetStream(shandle[i], streams[i]);
    cublasSetStream(chandle[i], streams[i]);
    cudaMalloc(reinterpret_cast<void**>(&workspace[i]), 16384);
    cublasSetWorkspace(chandle[i], workspace[i], 16384);
  }

#pragma omp parallel for num_threads(n_stream)
  for (int i = 0; i < d_lis.size(); i++) {
    int tid = omp_get_thread_num();

  }

  for (int i = 0; i < n_stream; i++) {
    cusolverDnDestroy(shandle[i]);
    cublasDestroy(chandle[i]);
    cudaFree(workspace[i]);
  }

}
