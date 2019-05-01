
#ifndef _CUDA_TIMER_CUH
#define _CUDA_TIMER_CUH

#include <pspl.cuh>

class event_linked_list
{
private:

  cudaEvent_t event;
  mark_t type;
  event_linked_list *next;

public:

  __host__ event_linked_list (const mark_t type_in, const cudaStream_t stream = 0)
  {
    cudaEventCreate (&event);
    cudaEventRecord (event, stream);
    type = type_in;
    next = nullptr;
  }

  __host__ ~event_linked_list ()
  {
    cudaEventDestroy(event);
    delete next;
  }

  __host__ void hookNewEvent (const mark_t type_in, const cudaStream_t stream = 0)
  {
    for (event_linked_list * ptr = this; ptr != nullptr; ptr = ptr -> next)
    { if (ptr -> next == nullptr) { ptr -> next = new event_linked_list (type_in, stream); return; } }
  }

  __host__ int length () const
  { 
    int l = 0;
    for (const event_linked_list * ptr = this; ptr != nullptr; ptr = ptr -> next) { l++; }
    return l;
  }

  __host__ int length (const mark_t type_in) const
  {
    int l = 0;
    for (const event_linked_list * ptr = this; ptr != nullptr; ptr = ptr -> next) 
    { if (ptr -> type == type_in) { l++; } }
    return l;
  }

  __host__ float getTotal_Sync (const event_linked_list *e = nullptr) const
  {
    float millis = 0.0;
    switch (type)
    {
    case start:
      millis = (next == nullptr) ? 0 : next -> getTotal_Sync (this);
      break;
    case end:
      if (e != nullptr) { cudaEventElapsedTime(&millis, e -> event, event); }
      millis += (next == nullptr) ? 0 : next -> getTotal_Sync (e);
      break;
    }
    return millis;
  }

};

class timer 
{
private:

  event_linked_list ** events;
  char ** names;
  int event_counter;
  int table_size;

public:

  __host__ timer (const int time_table_size = 16)
  {
    events = new event_linked_list * [time_table_size];
    names = new char * [time_table_size];
    memset ((void *) events, 0, time_table_size * sizeof(event_linked_list *));
    memset ((void *) names, 0, time_table_size * sizeof(char *));
    event_counter = 0;
    table_size = (time_table_size > 0) ? time_table_size : 1;
  }

  __host__ ~timer ()
  {
    for (int i = 0; i < event_counter; i++)
    { delete events[i]; delete names[i]; }
    delete[] events;
    delete[] names;
  }

  __host__ event_linked_list * getEvent (const char *event_name) const
  {
    for (int i = 0; i < event_counter; i++)
    {
      if (strcmp(event_name, names[i]) == 0) 
      { return events[i]; }
    }
    return nullptr;
  }

  __host__ void change_table_size (const int time_table_size)
  {
    int size = (time_table_size > 0) ? time_table_size : 1;

    event_linked_list ** events_old = events; 
    events = new event_linked_list *[size];
    memset((void *)events, 0, size * sizeof(event_linked_list *));

    char ** names_old = names;
    names = new char *[size];
    memset((void *)names, 0, size * sizeof(char *));

    for (int i = 0; i < event_counter && i < size; i++)
    {
      events[i] = events_old[i];
      names[i] = names_old[i];
    }
    for (int i = size; i < table_size; i++)
    {
      delete events_old[i];
      delete names_old[i];
    }
    delete[] events_old;
    delete[] names_old;

    table_size = size;
    event_counter = (event_counter > table_size) ? table_size : event_counter;
    printf("-- Timer: Table size changed to %d --\n\n", table_size);
  }

  __host__ void newEvent (const char *event_name, mark_t type, cudaStream_t stream = 0)
  {
    event_linked_list *p = getEvent(event_name);
    
    if (p == nullptr)
    {
      if (event_counter == table_size)
      {
        change_table_size(table_size * 2);
      }
      events[event_counter] = new event_linked_list(type, stream);
      const int length = (int) strlen(event_name) + 1;
      names[event_counter] = new char[length];
      strcpy(names[event_counter], event_name);
      event_counter ++;
    }
    else
    { p -> hookNewEvent(type, stream); }

  }

  __host__ double dumpAllEvents_Sync ()
  {
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) 
    { fprintf(stderr, "CUDA error from device synchronize: %s\n\n", cudaGetErrorString(error)); return 0.; }

    printf("-- Timer Summary --\nCUDA device synchronization success.\n");

    double accum = 0.;
    for (int i = 0; i < event_counter; i++)
    {
      float millis = events[i] -> getTotal_Sync();
      printf ("%s:  %f ms.\n", names[i], millis);
      accum += millis;

      delete events[i];
      delete[] names[i];
      events[i] = nullptr;
      names[i] = nullptr;
    }
    event_counter = 0;

    printf("All timed events cleared.\n\n");

    return accum;
  }

  __host__ void printStatus () const
  {
    printf("-- Timer Status: --\n"
      "Total Timed Events: %d.\n"
      "Time Table Size: %d.\n", event_counter, table_size);

    for (int i = 0; i < event_counter; i++)
    {
      printf("Event: %s has %d start and %d end marks.\n", 
        names[i], events[i] -> length(start), events[i] -> length(end));
    }

    printf("-- Timer Status End. --\n\n");
  }

};

#endif