#ifndef _INDEX_CUH
#define _INDEX_CUH

#include <stdio.h>

struct multi_level_index {

  int levels;
  int *ns;

  __host__ multi_level_index (const int levels_in = 0, const int *ns_in = nullptr, const int index_in = -1)
  {
    levels = ((levels_in > 0) ? levels_in : 0) + ((index_in >= 0) ? 1 : 0);
    if (levels > 0)
    {
      ns = new int [levels];
      for (int i = 0; i < levels - 1; i++) 
      { ns[i] = (ns_in == nullptr) ? -1 : ns_in[i]; }
      ns[levels - 1] = (index_in >= 0) ? index_in : ((ns_in == nullptr) ? -1 : ns_in[levels - 1]);
    }
    else
    { ns = nullptr; }
  }

  __host__ ~multi_level_index ()
  {
    delete[] ns;
  }

  __host__ void print () const
  {
    printf("-- ");
    if (levels == 0) printf("root");
    for(int i = 0; i < levels; i++)
    { printf("level %d: %d", i, ns[i]); if (i != levels - 1) printf(", "); }
    printf(" --\n");
  }

  __host__ void print_short () const
  {
    printf("%d", levels);
    for(int i = 0; i < levels; i++)
    { printf("%d", ns[i]); }
  }

  __host__ int compare (const struct multi_level_index *in) const
  {
    if (in == nullptr) { return -1; }

    int n = ((in -> levels) > levels) ? levels : (in -> levels);
    for (int i = 0; i < n; i++) 
    { if (ns[i] != (in -> ns)[i]) return -1; }

    if (in -> levels == levels)
    { return 0; }
    else
    { return (levels > n) ? ns[n] : (in -> ns)[n]; }
  }

};

#endif