/* This header file does nothing but to resolve mis-reported intellisense errors. */

#ifdef __INTELLISENSE__

#define __host__
#define __device__
#define atomicAdd
#define atomicSub
#define atomicExch
#define clock64() 0
#define __syncthreads()

#endif