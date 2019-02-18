
#ifndef _CUDA_TIMER_CUH
#define _CUDA_TIMER_CUH

#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <cuda.h>

/* Timer Functions */

struct timer {

  struct event_chain {
  
    cudaEvent_t event;
    struct event_chain *next;
  
    __host__ event_chain (cudaStream_t stream = 0)
    {
      cudaEventCreate (&event);
      cudaEventRecord (event, stream);
      next = nullptr;
    }
  
    __host__ ~event_chain ()
    {
      cudaEventDestroy (event);
      if (next != nullptr)
      { next -> ~event_chain(); free (next); }
    }
  
    __host__ struct event_chain * getLastChainElement ()
    {
      struct event_chain *p = this;
      while (p -> next != nullptr) { p = p -> next; }
      return p;
    }
  
    __host__ int length ()
    {
      return (next == nullptr) ? 1 : 1 + next -> length();
    }
  
  };

  struct event_chain **events;
  char **names;
  int event_counter;
  int table_size;

  __host__ timer (int time_table_size = 16)
  {
    events = (struct event_chain**) malloc (time_table_size * sizeof(struct event_chain*));
    names = (char **) malloc (time_table_size * sizeof(char *));
    memset ((void *) events, 0, time_table_size * sizeof(struct event_chain*));
    memset ((void *) names, 0, time_table_size * sizeof(char *));
    event_counter = 0;
    table_size = time_table_size;
  }

  __host__ ~timer ()
  {
    for (int i = 0; i < event_counter; i++)
    { events[i] -> ~event_chain(); free(events[i]); free(names[i]); }
    free (events);
    printf("-- Timer destructed. --\n\n");
  }

  __host__ void newEvent (const char *event_name, cudaStream_t stream = 0)
  {
    struct event_chain *p = nullptr;
    for (int i = 0; i < event_counter; i++)
    {
      if (strcmp(event_name, names[i]) == 0) { p = events[i]; }
    }

    if (p == nullptr && event_counter < table_size)
    {
      events[event_counter] = new event_chain(stream);
      names[event_counter] = (char *) malloc (strlen(event_name) * sizeof(char));
      strcpy(names[event_counter], event_name);
      event_counter ++;
    }
    else if (event_counter < table_size)
    { p -> getLastChainElement() -> next = new event_chain(stream); }
    else
    { printf("Table is full and Timer cannot add in another event: %s. \n\n", event_name); }

  }

  __host__ cudaError_t dumpAllEvents_Sync ()
  {
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) { return error; }

    printf("-----------------------------------------------------\n");
    printf("CUDA device synchronized, start dumping timed events:\n");

    for (int i = 0; i < event_counter; i++)
    {
      struct event_chain *e1 = events[i], *e2;

      float millis, total = 0.0;
      while(e1 != nullptr && e1 -> next != nullptr)
      {
        e2 = e1 -> next;
        cudaEventElapsedTime (&millis, e1 -> event, e2 -> event);
        total += millis;
        e1 = e2 -> next;
      }
      printf ("%s:  %f ms.\n", names[i], total);

      events[i] -> ~event_chain();
      free(events[i]);
      free(names[i]);
      events[i] = nullptr;
      names[i] = nullptr;
    }

    event_counter = 0;
    printf("All timed events dumped, table is cleared. \n");
    printf("-----------------------------------------------------\n\n");

    return cudaSuccess;
  }

  __host__ void printStatus ()
  {
    printf("-- Timer Status: --\nTotal Timed Events: %d.\n", event_counter);

    for (int i = 0; i < event_counter; i++)
    {
      printf("Event: %s has %d marks.\n", names[i], events[i] -> length());
    }

    printf("-- Timer Status End. --\n\n");
  }

};

#endif