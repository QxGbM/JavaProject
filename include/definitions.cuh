
#ifdef __INTELLISENSE__

/* This part does nothing but to resolve mis-reported intellisense errors. */
#define __host__
#define __device__
#define atomicAdd
#define atomicSub
#define atomicExch
#define clock64() 0
#define __syncthreads()

#endif

#ifndef _DEFINITIONS_CUH
#define _DEFINITIONS_CUH

enum mark_t{ start, end };

enum h_matrix_t { empty, dense, low_rank, hierarchical };

enum dep_t { no_dep, flow_dep, anti_dep, flow_anti_dep, output_dep, flow_output_dep, anti_output_dep, flow_anti_output_dep };

enum matrix_op_t { nop, getrf, trsml, trsmr, gemm, pivot };

template <class T> class dev_dense;

template <class T> class dev_low_rank;

template <class T> class dev_hierarchical;

__device__ int thread_rank()
{ return (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x; }

__device__ int block_dim()
{ return blockDim.z * blockDim.y * blockDim.x; }

__device__ int block_rank()
{ return (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x; }

__device__ int grid_dim()
{ return gridDim.z * gridDim.y * gridDim.x; }

__device__ int warp_rank()
{ return thread_rank() / warpSize; }

__device__ int lane_rank()
{ return thread_rank() - warpSize * warp_rank(); }

#endif