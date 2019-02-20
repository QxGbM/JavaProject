#ifndef _DAG_CUH
#define _DAG_CUH

#include <dev_hierarchical.cuh>

enum matrix_op_t {
  nop,
  getrf,
  gessm,
  tstrf,
  ssssm,
};

__host__ int calc_load (int op)
{
  int load_table[] = {1, 1, 1, 1, 1, 1, 1};
  return load_table[op];
}

struct ops_chain {

  matrix_op_t op_type;
  struct multi_level_index *dest;
  struct multi_level_index *m1;
  struct multi_level_index *m2;

  int load;
  struct ops_chain *next;
  struct ops_chain *child;

  __host__ ops_chain (matrix_op_t opin = nop, struct multi_level_index *in0 = nullptr, struct multi_level_index *in1 = nullptr, struct multi_level_index *in2 = nullptr)
  {
    op_type = opin;
    dest = (in0 == nullptr) ? nullptr : (in0 -> clone());
    m1 = (in1 == nullptr) ? nullptr : (in1 -> clone());
    m2 = (in2 == nullptr) ? nullptr : (in2 -> clone());

    load = calc_load ((int) opin);
    next = nullptr;
    child = nullptr;
  }

  __host__ ~ops_chain ()
  {
    if (dest != nullptr)
    { dest -> ~multi_level_index(); free(dest); }
    if (m1 != nullptr)
    { m1 -> ~multi_level_index(); free(m1); }
    if (m2 != nullptr)
    { m2 -> ~multi_level_index(); free(m2); }
    if (next != nullptr)
    { next -> ~ops_chain(); free(next); }
    if (child != nullptr)
    { child -> ~ops_chain(); free(child); }
  }

  __host__ void hookup (struct ops_chain *chain)
  {
    if (next != nullptr)
    { next -> hookup(chain); }
    else
    { next = chain; }
  }

  __host__ struct ops_chain * lookup (const int index)
  {
    if (child == nullptr)
    {
      if (index == 0) { return this; }
      if (next != nullptr) { return next -> lookup(index - 1); } 
    }

    if (child != nullptr)
    {
      int length = child -> length();
      if (index < length) { return child -> lookup(index); }
      if (next != nullptr) { return next -> lookup(index - length); }
    }

    return nullptr;
  }

  __host__ int length ()
  {
    int l_child = (child == nullptr) ? 1 : child -> length();
    int l_next = (next == nullptr) ? 0 : next -> length();
    return l_child + l_next;
  }

  __host__ void print (const int op_id = 0, const bool indent = true, const bool recurse = true)
  {
    for (int i = 0; i < (dest -> levels) && indent; i++) { printf("  "); }

    if (child == nullptr) { printf("%d: ", op_id); }
    switch(op_type)
    {
      case nop: printf("NOP "); break;
      case getrf: printf("GETRF "); break;
      case gessm: printf("GESSM "); break;
      case tstrf: printf("TSTRF "); break;
      case ssssm: printf("SSSSM "); break;
    }

    if (dest != nullptr) { dest -> print_short(); printf(", "); }
    else { printf("_, "); }
    if (m1 != nullptr) { m1 -> print_short(); printf(", "); }
    else { printf("_, "); }
    if (m2 != nullptr) { m2 -> print_short(); }
    else { printf("_"); }
    printf("\n");

    if (child != nullptr && recurse) { child -> print(op_id, indent, recurse); }

    int l_child = (child == nullptr) ? 1 : child -> length();
    if (next != nullptr && recurse) { next -> print(op_id + l_child, indent, recurse); }

    if ((next == nullptr && dest -> levels == 1) || !recurse) { printf("\n"); }
  }

};

template <class matrixEntriesT> __host__ struct ops_chain * get_ops_hgetrf (const struct dev_hierarchical <matrixEntriesT> *a)
{
  struct ops_chain *ops = nullptr;
  int nx = a -> nx, ny = a -> ny, n = (nx > ny) ? ny : nx;
  for (int i = 0; i < n; i++)
  {
    struct h_matrix_element <matrixEntriesT> *e0 = (a -> elements)[i * nx + i], *e1, *e2;
    struct ops_chain *p0 = new ops_chain(getrf, e0 -> index), *p1;
    if (e0 -> element_type == hierarchical) 
    { p0 -> child = get_ops_hgetrf ((struct dev_hierarchical <matrixEntriesT> *) (e0 -> element)); }

    for (int j = i + 1; j < nx; j++)
    {
      e1 = (a -> elements)[i * nx + j];
      p1 = new ops_chain(gessm, e1 -> index, e0 -> index);
      // TODO: hgessm
      //if (e1 -> element_type == hierarchical) 
      //{ p1 -> child = get_ops_htrsm ((struct dev_hierarchical <matrixEntriesT> *) (e1 -> element)); }
      p0 -> hookup(p1);
    }

    for (int j = i + 1; j < ny; j++)
    {
      e2 = (a -> elements)[j * nx + i];
      p1 = new ops_chain(tstrf, e2 -> index, e0 -> index);
      // TODO: htstrf
      //if (e2 -> element_type == hierarchical) 
      //{ p1 -> child = get_ops_htrsm ((struct dev_hierarchical <matrixEntriesT> *) (e2 -> element)); }
      p0 -> hookup(p1);
    }

    for (int j = i + 1; j < ny; j++)
    {
      for (int k = i + 1; k < nx; k++)
      {
        e0 = (a -> elements)[j * nx + k];
        e1 = (a -> elements)[j * nx + i];
        e2 = (a -> elements)[i * nx + k];
        p1 = new ops_chain(ssssm, e0 -> index, e1 -> index, e2 -> index);
        // TODO: hgemm 
        //if (e2 -> element_type == hierarchical) 
        //{ p1 -> child = get_ops_hgemm ((struct dev_hierarchical <matrixEntriesT> *) (e0 -> element)); }
        p0 -> hookup(p1);
      }
    }

    if (ops == nullptr) { ops = p0; }
    else { ops -> hookup(p0); }
  }
  return ops;
}

enum dep_t {
  no_dep,
  flow_dep,
  anti_dep,
  flow_anti_dep,
  output_dep,
  flow_output_dep,
  anti_output_dep,
  flow_anti_output_dep,
};

__host__ dep_t add_dep (dep_t x, dep_t y) {
  int a = (int) x, b = (int) y, r = 0;
  for (int i = 4; i > 0; i = i / 2)
  {
    if (a >= i || b >= i) 
    { 
      r += i; 
      a = (a >= i) ? a - i : a;
      b = (b >= i) ? b - i : b; 
    }
  }
  return (dep_t) r; 
}

__host__ char * dep_str (dep_t x) {
  int a = (int) x;
  char *str = (char *) malloc (4 * sizeof(char));
  if (a == 0) { sprintf(str, "  ND"); return str; }

  if (a >= 4) { str[2] = 'O'; a -= 4; } else { str[2] = ' '; }
  if (a >= 2) { str[1] = 'A'; a -= 2; } else { str[1] = ' '; }
  if (a >= 1) { str[0] = 'F'; a -= 1; } else { str[0] = ' '; }
  str[3] = 'D';

  if (str[2] == ' ') 
  { 
    if (str[1] == 'A') { str[2] = 'A'; str[1] = str[0]; str[0] = ' '; }
    else { str[2] = 'F'; str[0] = ' '; }
  }
  else if (str[1] == ' ' && str[0] == 'F')
  { str[1] = 'F'; str[0] = ' '; }

  return str;
}


struct dag {

  struct ops_chain *ops;

  int length;
  dep_t *dep;

  __host__ dag (struct ops_chain * chain) {
    ops = chain;
    length = chain -> length();
    dep = (dep_t *) malloc (length * length * sizeof(dep_t));
    memset ((void *) dep, 0, length * length * sizeof(dep_t));
    build_dep();
  }

  __host__ ~dag ()
  {
    ops -> ~ops_chain();
    free(ops);
    free(dep);
  }

  __host__ void build_dep()
  {
    for (int i = 0; i < length; i++)
    {
      struct ops_chain *op_src = ops -> lookup(i);
      struct multi_level_index *dest_src = op_src -> dest, *m1_src = op_src -> m1, *m2_src = op_src -> m2;

      for (int j = i + 1; j < length; j++)
      {
        struct ops_chain *op = ops -> lookup(j);
        struct multi_level_index *dest = op -> dest, *m1 = op -> m1, *m2 = op -> m2;

        if (dest_src != nullptr) 
        {
          if (dest_src -> compare(m1) >= 0)
          { dep[j * length + i] = add_dep(flow_dep, dep[j * length + i]); }
          if (dest_src -> compare(m2) >= 0)
          { dep[j * length + i] = add_dep(flow_dep, dep[j * length + i]); }
          if (dest_src -> compare(dest) >= 0)
          { dep[j * length + i] = add_dep(output_dep, dep[j * length + i]); }
        }

        if (dest != nullptr)
        {
          if (dest -> compare(m1_src) >= 0)
          { dep[j * length + i] = add_dep(anti_dep, dep[j * length + i]); }
          if (dest -> compare(m2_src) >= 0)
          { dep[j * length + i] = add_dep(anti_dep, dep[j * length + i]); }
        }
      }
    }
  }

  __host__ void print()
  {
    ops -> print();
    for (int i = 0; i < length; i++)
    {
      printf("%d:\t", i);
      for (int j = 0; j < length; j++)
      {
        if (j > i)
        {
          char *str = dep_str(dep[j * length + i]); 
          printf("%s ", str);
          free(str);
        }
        else
        { printf("     "); }
      }
      printf("\n");
    }
    printf("\n");
  }
  
};


#endif