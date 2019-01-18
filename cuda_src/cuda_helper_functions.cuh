#ifndef _CUDA_HELPER
#define _CUDA_HELPER

#include <stdio.h>
#include <string.h>
#include <cuda.h>

#define TIME_TABLE_SIZE 16

__host__ cudaError_t matrix_copy_toDevice_sync (double *matrix, double **dev_matrix, const unsigned nx, const unsigned ld, const unsigned ny);

__host__ cudaError_t matrix_copy_toHost_sync (double **dev_matrix, double *matrix, const unsigned nx, const unsigned ld, const unsigned ny, const bool free_device);

__host__ cudaError_t create_timing_event_to_stream (const char* event_name, cudaStream_t stream);

__host__ cudaError_t device_sync_dump_timed_events ();

#endif