
/* This part does nothing but to resolve mis-reported intellisense errors. */
#ifdef __INTELLISENSE__

#define __syncthreads()
#define asm
#define volatile()

#endif

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <memory.h>
#include <time.h>

#include <omp.h>
#include <cub/cub.cuh>

#ifndef _PSPL_CUH
#define _PSPL_CUH

#define _MAX_INST_LENGTH 32
#define _MAX_SCHEDULER_ITERS 1024

#define _DEFAULT_PTRS_LENGTH 1024
#define _DEFAULT_INSTS_LENGTH 1024
#define _DEFAULT_COMM_LENGTH 1024
#define _DEFAULT_COMPRESSOR_LENGTH 1024

#define _RND_SEED_LENGTH 8192

enum mark_t { start, end };

enum element_t { empty, dense, low_rank, hierarchical, temp_dense, temp_low_rank };

enum dependency_t { no_dep, flow_dep, anti_dep, flow_anti_dep, output_dep, flow_output_dep, anti_output_dep, flow_anti_output_dep };

enum operation_t { nop, getrf, trsml, trsmr, accum, pivot, gemm, gemm_plus, gemm_3x, gemm_4x, accum_dense };

enum relation_t { diff_mat, same_mat_diff_branch, same_branch_diff_node, same_node_no_overlap, same_node_overlapped, same_node_different_temp, same_index };

enum opcode_t { execute, signal_wait, finish };

__constant__ double dev_rnd_seed [_RND_SEED_LENGTH];

__device__ __forceinline__ int thread_rank()
{ return (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x; }

__device__ __forceinline__ int block_dim()
{ return blockDim.z * blockDim.y * blockDim.x; }

__device__ __forceinline__ int block_rank()
{ return (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x; }

__device__ __forceinline__ int grid_dim()
{ return gridDim.z * gridDim.y * gridDim.x; }

__device__ __forceinline__ int warp_rank()
{
  unsigned int warpid;
  asm volatile("mov.u32 %0, %warpid;" : "=r"(warpid));
  return (int) warpid;
}

__device__ __forceinline__ int lane_rank()
{ 
  unsigned int laneid;
  asm volatile("mov.u32 %0, %laneid;" : "=r"(laneid));
  return (int) laneid;
}

__device__ __forceinline__ int num_warps()
{ return (block_dim() + warpSize - 1) / warpSize; }

template <class T> class dev_dense;
template <class T> class dev_low_rank;
template <class T> class dev_hierarchical;
template <class T> class dev_h_element;
class dev_temp;

class h_index;
class h_ops;
class h_ops_tree;
class h_ops_dag;

class instructions_queue;
class instructions_scheduler;

class instructions_manager;

class dependency_linked_list;
class event_linked_list;
class timer;
class compressor;

#include <dev_hierarchical_index.cuh>
#include <dev_hierarchical_ops.cuh>

#include <dev_dense_funcs.cuh>
#include <dev_low_rank_funcs.cuh>

#include <dev_temp.cuh>
#include <dev_dense.cuh>
#include <compressor.cuh>
#include <dev_low_rank.cuh>
#include <dev_hierarchical.cuh>
#include <dev_hierarchical_element.cuh>

#include <dev_hierarchical_ops_dag.cuh>

#include <instructions_scheduler.cuh>
#include <instructions_manager.cuh>

#include <timer.cuh>
#include <kernel.cuh>


#endif