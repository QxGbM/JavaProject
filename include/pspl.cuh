
/* This part does nothing but to resolve mis-reported intellisense errors. */
#ifdef __INTELLISENSE__

#define __host__
#define __device__
#define __restrict__
#define __launch_bounds__()
#define atomicAdd
#define atomicSub
#define atomicExch
#define clock64() 0
#define __syncthreads()
#define __syncwarp()

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

#ifndef _PSPL_CUH
#define _PSPL_CUH

#define _MAX_INST_LENGTH 32
#define _MAX_SCHEDULER_ITERS 1024
#define _RND_SEED_LENGTH 8192

enum mark_t{ start, end };

enum element_t { empty, dense, low_rank, hierarchical };

enum dependency_t { no_dep, flow_dep, anti_dep, flow_anti_dep, output_dep, flow_output_dep, anti_output_dep, flow_anti_output_dep };

enum operation_t { nop, getrf_d, 
  trsml_d, trsml_lr, trsmr_d, trsmr_lr, pivot_d, pivot_lr,
  gemm_d_d_d, gemm_d_d_lr, gemm_d_lr_d, gemm_d_lr_lr, gemm_lr_d_d, gemm_lr_d_lr, gemm_lr_lr_d, gemm_lr_lr_lr };

enum relation_t { diff_mat, same_mat_diff_branch, same_branch_diff_node, same_node_no_overlap, same_node_overlapped, same_index };

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

template <class T> class dev_dense;
template <class T> class dev_low_rank;
template <class T> class dev_hierarchical;
template <class T> class dev_h_element;

class h_index;
class h_ops;
class h_ops_tree;
class h_ops_dag;

class inst_queue;
class inst_scheduler;

template <class T> class dev_instructions;

class dependency_linked_list;
class event_linked_list;
class timer;

#include <timer.cuh>

#include <dev_hierarchical_index.cuh>
#include <dev_hierarchical_ops.cuh>
#include <dev_hierarchical_ops_dag.cuh>

#include <dev_dense.cuh>
#include <dev_dense_funcs.cuh>

#include <dev_low_rank.cuh>
#include <dev_low_rank_funcs.cuh>

#include <dev_hierarchical.cuh>
#include <dev_hierarchical_element.cuh>

#include <inst_scheduler.cuh>
#include <dev_instructions.cuh>
#include <kernel.cuh>

#endif