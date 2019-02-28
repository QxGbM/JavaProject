#ifndef _INDEX_CUH
#define _INDEX_CUH

struct multi_level_index {

  int levels;
  int *ns;

  __host__ multi_level_index (const int l, const int *n = nullptr)
  {
    levels = (l > 0) ? l : 1;
    ns = (int *) malloc (levels * sizeof(int));
    memset(ns, 0, levels * sizeof(int));

    for(int i = 0; i < levels && l > 0 && n != nullptr; i++) 
    { ns[i] = n[i]; }
  }

  __host__ multi_level_index (const int n, const struct multi_level_index *parent = nullptr)
  {
    levels = (parent == nullptr) ? 1 : (1 + parent -> levels);
    ns = (int *) malloc (levels * sizeof(int));

    for(int i = 0; i < levels - 1 && parent != nullptr; i++) 
    { ns[i] = (parent -> ns)[i]; }
    ns[levels - 1] = n;
  }

  __host__ ~multi_level_index ()
  {
    free(ns);
  }

  __host__ void print ()
  {
    printf("-- ");
    for(int i = 0; i < levels; i++)
    { printf("level %d: %d", i, ns[i]); if (i != levels - 1) printf(", "); }
    printf(" --\n");
  }

  __host__ struct multi_level_index * clone()
  {
    struct multi_level_index *p = (struct multi_level_index *) malloc (sizeof(struct multi_level_index));
    p -> levels = levels;
    p -> ns = (int *) malloc (levels * sizeof(int));
    for (int i = 0; i < levels; i++) { (p -> ns)[i] = ns[i]; }
    return p;
  }

  __host__ void print_short ()
  {
    printf("%d", levels);
    for(int i = 0; i < levels; i++)
    { printf("%d", ns[i]); }
  }

  __host__ int compare (const struct multi_level_index *in)
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