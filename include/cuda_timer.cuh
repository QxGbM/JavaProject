#ifndef CUDA_TIMER_CUH
#define CUDA_TIMER_CUH

#include <cuda.h>
#define TIME_TABLE_SIZE 16
#define MAX_NAME_LENGTH 16

/* Timer Functions */

struct event_chain {
  cudaEvent_t event;
  char* name;
  struct event_chain *next;
};

struct event_chain **events = nullptr;
int event_counter = 0;

__host__ cudaError_t create_timing_event_to_stream (const char* event_name, cudaStream_t stream = 0)
{
  cudaError_t error = cudaSuccess;
  if (events == nullptr) 
  { 
    events = (struct event_chain**) malloc(TIME_TABLE_SIZE * sizeof(struct event_chain*));
    for (unsigned i = 0; i < TIME_TABLE_SIZE; i++) { events[i] = nullptr; }
  }

  struct event_chain *p = nullptr;
  for (int i = 0; i < event_counter; i++)
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
  p -> name = (char*) malloc(MAX_NAME_LENGTH * sizeof(char));
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

  for (int i = 0; i < event_counter; i++)
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

#endif