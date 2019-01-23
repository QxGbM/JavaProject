
#ifndef _CUDA_HELPER_CUH
#define _CUDA_HELPER_CUH

#include <stdio.h>
#include <string.h>
#include <cuda.h>

#define BLOCK_SIZE 16 // Each block has 16^2 = 1024 threads, make sure the cuda device allows
#define TIME_TABLE_SIZE 16
#define CUDA_DEVICE 0

__host__ cudaError_t matrix_copy_toDevice_sync (double *matrix, double **dev_matrix, const unsigned nx, const unsigned ld, const unsigned ny)
{
  /* 
  * A synchronous copy of matrix to dev_matrix.
  * Matrix can be stored in pageable memory.
  * This function also allocates dev_matrix if it is not allocated.
  * NOTE: The device matrix need to be aligned with BLOCK_SIZE, which means ld and ny are multiples of BLOCK_SIZE
  */
  if (ld < nx) { printf("MEM COPY ABORT: Matrix x's horizontal offset is less than the number of entries.\n");  return cudaErrorInvalidConfiguration; }

  cudaError_t error = cudaSuccess;
  const unsigned ld_aligned = ((ld + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
  const unsigned ny_aligned = ((ny + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
  
  if (*dev_matrix == 0) 
  {
    if (ld_aligned != ld || ny_aligned != ny) 
    { 
      printf("WARNING: Input matrix's dimensions [%d x %d offset %d x %d] are not aligned with block size.\n", ny, nx, ny, ld);
      printf("-------- Automatically align it to: [%d x %d offset %d x %d].\n\n", ny, nx, ny_aligned, ld_aligned); 
    }
    error = cudaMalloc((void**)dev_matrix, ld_aligned * ny_aligned * sizeof(double));
    if (error != cudaSuccess) { return error; }

    printf("Allocated %d x %d matrix in cuda global memory.\n\n", ny_aligned, ld_aligned);
  }
  else if (ld_aligned != ld || ny_aligned != ny) 
  {
    printf("WARNING: Device Matrix might not have been auto-aligned with block size.\n");
    printf("-------- Please confirm dev-matrix is allocated properly.\n");
    printf("-------- Or, let this matrix-copy function allocate dev-matrix.\n\n");
  }

  for (unsigned i = 0; i < ny; i++)
  {
    error = cudaMemcpy(&(*dev_matrix)[i * ld_aligned], &matrix[i * ld], nx * sizeof(double), cudaMemcpyHostToDevice);
    if (error != cudaSuccess) { return error; }
  }
  
  printf("Copied %d x %d entries from host to cuda global memory.\n\n", ny, nx);
  return cudaSuccess;
}

__host__ cudaError_t matrix_copy_toHost_sync (double **dev_matrix, double *matrix, const unsigned nx, const unsigned ld, const unsigned ny, const bool free_device)
{
  /* 
  * A synchronous copy of dev_matrix to matrix
  * Matrix can be stored in pageable memory
  */
  if (ld < nx) { printf("MEM COPY ABORT: Matrix x's horizontal offset is less than the number of entries.\n");  return cudaErrorInvalidConfiguration; }

  cudaError_t error = cudaSuccess;
  const unsigned ld_aligned = ((ld + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;
  const unsigned ny_aligned = ((ny + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

  for (unsigned i = 0; i < ny; i++)
  {
    error = cudaMemcpy(&matrix[i * ld], &(*dev_matrix)[i * ld_aligned], nx * sizeof(double), cudaMemcpyDeviceToHost);
    if (error != cudaSuccess) { return error; }
  }
  
  printf("Copied %d x %d entries from cuda global memory to host.\n\n", ny, nx);

  if (free_device)
  {
    error = cudaFree(dev_matrix);
    if (error != cudaSuccess) { return error; }

    printf("Freed %d x %d matrix in cuda global memory.\n\n", ny_aligned, ld_aligned);
  }

  return cudaSuccess;
}

/* Timer Functions */

struct event_chain {
  cudaEvent_t event;
  char* name;
  struct event_chain *next;
};

struct event_chain **events = nullptr;
unsigned event_counter = 0;

__host__ cudaError_t create_timing_event_to_stream (const char* event_name, cudaStream_t stream)
{
  cudaError_t error = cudaSuccess;
  if (events == nullptr) 
  { 
    events = (struct event_chain**) malloc(TIME_TABLE_SIZE * sizeof(struct event_chain*));
    for (unsigned i = 0; i < TIME_TABLE_SIZE; i++) { events[i] = nullptr; }
  }

  struct event_chain *p = nullptr;
  for (unsigned i = 0; i < event_counter; i++)
  {
    if ((events[i] != nullptr) && (strcmp(event_name, events[i] -> name) == 0)) { p = events[i]; }
  }

  if (p == nullptr)
  {
    p = (struct event_chain*) malloc(sizeof(struct event_chain));
    events[event_counter] = p;
    event_counter++;
  }
  else 
  {
    while (p -> next != nullptr) { p = p -> next; }
    p -> next = (struct event_chain*) malloc(sizeof(struct event_chain));
    p = p -> next;
  }

  error = cudaEventCreate(&(p -> event));
  p -> name = (char*) malloc(16 * sizeof(char));
  strcpy(p -> name, event_name);
  p -> next = nullptr;
  error = cudaEventRecord(p -> event, stream);

  return error;

}

__host__ cudaError_t device_sync_dump_timed_events ()
{
  cudaError_t error = cudaDeviceSynchronize();
  if (error != cudaSuccess) { return error; }

  printf("--------------------------------------------------------\n");
  printf("All CUDA execution finished, start dumping timed events:\n");

  for (unsigned i = 0; i < event_counter; i++)
  {
    struct event_chain *e1 = events[i], *e2;
    char *name = (char*) malloc(16 * sizeof(char));
    strcpy(name, e1 -> name);
    float millis, total = 0.0;
    while(e1 != nullptr && e1 -> next != nullptr)
    {
      e2 = e1 -> next;
      cudaEventElapsedTime(&millis, e1 -> event, e2 -> event);
      total += millis;
      e1 = e2 -> next;
    }
    printf ("%s:  %f ms.\n", name, total);

    e1 = events[i];
    while(e1 != nullptr)
    {
      e2 = e1 -> next;
      cudaEventDestroy(e1 -> event);
      free(e1 -> name);
      free(e1);
      e1 = e2;
    }
    free(name);
  }

  event_counter = 0;
  printf("All timed events dumped, table is cleared. \n");
  printf("--------------------------------------------------------\n\n");

  return cudaSuccess;
}

/* bit-shifting offset handling */

struct offset {

  unsigned ld_bits;
  unsigned ld_value;

  offset(const unsigned ld)
  { ld_bits = 0; ld_value = ld; } //TODO
};

#endif