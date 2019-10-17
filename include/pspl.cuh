
/* This part does nothing but to resolve mis-reported intellisense errors. */
#ifdef __INTELLISENSE__

#define __syncthreads()
#define __threadfence()
#define __syncwarp()
#define asm
#define volatile()
#define clock64() 0
#define rsqrt() 0
#define rhypot() 0
#define __shfl_sync() 0
#define __shfl_xor_sync() 0

#endif

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <memory.h>
#include <time.h>

#include <omp.h>

#pragma once
#ifndef _PSPL_CUH
#define _PSPL_CUH

#ifdef _PSPL_USE_SINGLE
typedef float real_t;
typedef float4 vec_t;
const int vec_size = 4;
const int real_bits = 4;
#else
typedef double real_t;
typedef double2 vec_t;
const int vec_size = 2;
const int real_bits = 8;
#endif

//#define _PSPL_DEVICE_INLINE
#ifdef _PSPL_DEVICE_INLINE
#define DEVICE __device__ __forceinline__
#else
#define DEVICE __device__
#endif

#define _SHM_SIZE 12288
#define _MAX_INST_LENGTH 32
#define _MIN_INST_FLOPS 1000000
#define _SHADOW_RANK 16
#define _PTRS_LENGTH 1024
#define _INSTS_LENGTH 1024
#define _COMM_LENGTH 1024
#define _COMPRESSOR_LENGTH 1024
#define _BLOCK_M 64
#define _BLOCK_K 16
#define _CLOCK_MULTIPLIER 1.e-3
#define _SEED 200
#define _TICKS 500000
#define _ROW_BLOCKS 80

#define abs(x) ((x)<0 ? -(x) : (x))

enum mark_t { start, end };

enum element_t { empty, dense, low_rank, hierarchical, temp_dense, temp_low_rank, shadow };

enum dependency_t { no_dep, flow_dep, anti_dep, flow_anti_dep, output_dep, flow_output_dep, anti_output_dep, flow_anti_output_dep };

enum operation_t { nop, getrf, trsml, trsmr, gemm, gemm_plus, gemm_3x, gemm_4x, accum, accum_dense, pivot };

enum relation_t { diff_mat, same_mat_diff_branch, same_branch_diff_node, same_node_no_overlap, same_node_overlapped, same_node_different_temp, same_index };

enum opcode_t { execute, signal_wait, finish };

enum operation_length { nop_l = 3, getrf_l = 8, trsml_l = 13, trsmr_l = 13, gemm_l = 17, gemm_plus_l = 17, gemm_3x_l = 23, gemm_4x_l = 29, accum_l = 21, accum_dense_l = -1, pivot_l = -1 };


class dev_dense;
class dev_low_rank;
class dev_hierarchical;
class dev_h_element;
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

#include <dev_temp.cuh>
#include <matrix/dev_dense.cuh>
//#include <compressor.cuh>
#include <matrix/dev_low_rank.cuh>
#include <matrix/dev_hierarchical.cuh>
#include <matrix/dev_hierarchical_element.cuh>

#include <h_ops/dev_hierarchical_index.cuh>
#include <h_ops/dev_hierarchical_ops.cuh>
#include <h_ops/dev_hierarchical_ops_tree.cuh>
#include <h_ops/dev_hierarchical_ops_dag.cuh>

#include <instructions_scheduler.cuh>
#include <instructions_manager.cuh>

//#include <timer.cuh>
//#include <dev_dense_funcs.cuh>
//#include <dev_low_rank_funcs.cuh>
//#include <kernel.cuh>


#endif