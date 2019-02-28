
#ifndef _OPS_CUH
#define _OPS_CUH

#include <index.cuh>

enum matrix_op_t {
  nop,
  getrf,
  gessm,
  tstrf,
  ssssm,
};

__host__ int calc_load (int op) {
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

#endif