
#include <matrix/Compressor.cuh>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cublas_v2.h>
#include <cusolverDn.h>
#include <omp.h>

const int n_stream = 4;

compressor::compressor(Hierarchical& h, const int rank, const double condition) {
  load(h, rank, condition);
  compress(rank);
}

compressor::~compressor() {
  for (auto iter = d_lis.begin(); iter != d_lis.end(); iter++) {
    delete *iter;
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

void compressor::compress(const int rank) {
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

  int size = (int)d_lis.size() / n_stream;
#pragma omp parallel num_threads(n_stream) 
  {
    int tid = omp_get_thread_num();
    int start_i = tid * size;
    using std::min;
    int end_i = min((int)d_lis.size(), start_i + size);

    for (int i = start_i; i < end_i; i++) {
      LowRank* lr = lr_lis[i]->getElementLowRank();
      real_t* Q = lr->getU()->getElements();
    }
#pragma omp critical
    std::cout << tid << ": " << start_i << " " << end_i << std::endl;
  }

  for (int i = 0; i < n_stream; i++) {
    cusolverDnDestroy(shandle[i]);
    cublasDestroy(chandle[i]);
    cudaFree(workspace[i]);
  }

}
