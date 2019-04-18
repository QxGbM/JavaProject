
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
    if (next != nullptr)
    { delete next; }
  }

  __host__ void hookNewEvent (const mark_t type_in, const cudaStream_t stream = 0)
  {
    if (next == nullptr)
    { next = new event_linked_list (type_in, stream); }
    else
    { next -> hookNewEvent(type_in, stream); }
  }

  __host__ int length () const
  { return (next == nullptr) ? 1 : 1 + next->length(); }

  __host__ int length (const mark_t type_in) const
  {
    int count = (int) (type == type_in);
    return (next == nullptr) ? count : count + next -> length (type_in);
  }

  __host__ float getTotal_Sync (const event_linked_list *e = nullptr) const
  {
    float millis = 0.0;
    switch (type)
    {
    case start:
      millis = (next == nullptr) ? 0 : next->getTotal_Sync(this);
      break;
    case end:
      if (e != nullptr) { cudaEventElapsedTime(&millis, e -> event, event); }
      millis += (next == nullptr) ? 0 : next->getTotal_Sync(e);
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

  __host__ timer (const int time_table_size = 1)
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
      if (strcmp(event_name, names[i]) == 0) { return events[i]; }
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

  __host__ cudaError_t dumpAllEvents_Sync (const unsigned long long int total_fops = 0)
  {
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) 
    { fprintf(stderr, "CUDA error from device synchronize: %s\n\n", cudaGetErrorString(error)); return error; }

    printf("-----------------------------------------------------\n");
    printf("CUDA device synchronized, start dumping timed events:\n");

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

    if (total_fops > 0)
    { 
      double flops = 1.e3 * total_fops / accum;
      int power = 0;
      while (power < 4 && flops > 1.e3) { flops *= 1.e-3; power ++; }

      printf("# of float-point OPs: %llu \nTotal FLOPS: %f ", total_fops, flops);
      switch (power)
      {
      case 0: break;
      case 1: printf("K"); break;
      case 2: printf("M"); break;
      case 3: printf("G"); break;
      case 4: printf("T"); break;
      }
      printf("FLOPS. \n");
    }
    printf("All timed events dumped, table is cleared.\n");
    printf("-----------------------------------------------------\n\n");

    return cudaSuccess;
  }

  __host__ void printStatus () const
  {
    printf("-- Timer Status: --\n");
    printf("Total Timed Events: %d.\n", event_counter);
    printf("Time Table Size: %d.\n", table_size);

    for (int i = 0; i < event_counter; i++)
    {
      printf("Event: %s has %d start and %d end marks.\n", names[i], events[i] -> length(start), events[i] -> length(end));
    }

    printf("-- Timer Status End. --\n\n");
  }

};

#endif