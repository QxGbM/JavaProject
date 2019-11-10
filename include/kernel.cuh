
#pragma once
#ifndef _KERNEL_CUH
#define _KERNEL_CUH

#include <definitions.cuh>

__global__ void kernel_dynamic(const int** __restrict__ insts, void** __restrict__ ptrs, volatile int* __restrict__ comm_space,
  real_t** __restrict__ block_tmps, real_t* __restrict__ dev_rnd_seed, unsigned long long** __restrict__ clocks);



#endif