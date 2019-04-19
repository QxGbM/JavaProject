
/* This part does nothing but to resolve mis-reported intellisense errors. */
#ifdef __INTELLISENSE__

#define __host__
#define __device__
#define atomicAdd
#define atomicSub
#define atomicExch
#define clock64() 0
#define __syncthreads()

#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <math.h>
#include <string.h>
#include <memory.h>

#ifndef _PSPL_CUH
#define _PSPL_CUH

#define MAX_WARPS 32

enum mark_t{ start, end };

enum element_t { empty, dense, low_rank, hierarchical };

enum dependency_t { no_dep, flow_dep, anti_dep, flow_anti_dep, output_dep, flow_output_dep, anti_output_dep, flow_anti_output_dep };

enum operation_t { nop, getrf, trsml, trsmr, gemm, pivot, trsml_lr, trsmr_lr, gemm3, gemm4, gemm5 };

enum relation_t { diff_matrix, no_relation, diff_offset_no_overlap, diff_offset_overlapped, same_index, contains, contained };

enum opcode_t { execute, signal_wait, signal_write, finish };

__device__ inline int thread_rank()
{ return (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x; }

__device__ inline int block_dim()
{ return blockDim.z * blockDim.y * blockDim.x; }

__device__ inline int block_rank()
{ return (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x; }

__device__ inline int grid_dim()
{ return gridDim.z * gridDim.y * gridDim.x; }

__device__ inline int warp_rank()
{ return thread_rank() / warpSize; }

__device__ inline int lane_rank()
{ return thread_rank() - warpSize * warp_rank(); }

__device__ inline int num_warps()
{ return (block_dim() + warpSize - 1) / warpSize; }

#include <timer.cuh>

#include <dev_dense.cuh>
#include <dev_dense_funcs.cuh>

#include <dev_low_rank.cuh>
#include <dev_low_rank_funcs.cuh>

class h_index;
class h_ops_tree;
class h_ops;

template <class T> class dev_hierarchical;
template <class T> class dev_h_element;

#include <dev_hierarchical_index.cuh>
#include <dev_hierarchical_ops.cuh>
#include <dev_hierarchical_ops_dag.cuh>
#include <dev_hierarchical.cuh>
#include <dev_hierarchical_element.cuh>

#include <inst_scheduler.cuh>
#include <dev_instructions.cuh>
#include <kernel.cuh>

#endif