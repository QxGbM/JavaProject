
#pragma once
#ifndef _LAUNCHER_CUH
#define _LAUNCHER_CUH

#include <definitions.cuh>

void print_dev_mat (real_t * dev_mat, const int nx, const int ny);

cudaError_t allocate_clocks (unsigned long long *** clocks, const int workers, const int * lengths);

cudaError_t generateLaunchArgsFromTree (int *** dev_insts, void *** dev_ptrs, int ** comm_space, real_t *** block_tmps, real_t ** dev_rnd_seed, unsigned long long *** clocks,
  instructions_scheduler ** schedule_addr, double * total_lapse, long long * flops, const h_ops_tree * tree, real_t ** tmp_ptrs, const int workers, const int start_index = 0, const int length_max = 0);

cudaError_t launchKernelWithArgs (int ** dev_insts, void ** dev_ptrs, int * comm_space, real_t ** block_tmps, real_t * dev_rnd_seed, unsigned long long ** clocks, 
  const int workers, const int num_threads, cudaStream_t main_stream = 0);

cudaError_t dev_hierarchical_GETRF (Hierarchical * h, const int num_blocks, const int num_threads, const int kernel_size = 0);

#endif