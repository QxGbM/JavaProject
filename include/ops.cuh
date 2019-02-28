
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

  int n_read_write;
  struct multi_level_index **m_read_write;

  int n_read_only;
  struct multi_level_index **m_read_only;

  int load;
  struct ops_chain *next;
  struct ops_chain *child;

  __host__ ops_chain (matrix_op_t opin = nop, int n_read_write_in = 0, struct multi_level_index **in0 = nullptr, 
    int n_read_only_in = 0, struct multi_level_index **in1 = nullptr)
  {
    op_type = opin;

    n_read_write = n_read_write_in;
    m_read_write = (struct multi_level_index **) malloc (n_read_write * sizeof(struct multi_level_index *));
    for(int i = 0; i < n_read_write; i++) { m_read_write[i] = in0[i]; }

    n_read_only = n_read_only_in;
    m_read_only = (struct multi_level_index **) malloc (n_read_only * sizeof(struct multi_level_index *));
    for(int i = 0; i < n_read_only; i++) { m_read_only[i] = in1[i]; }

    load = calc_load ((int) opin);
    next = nullptr;
    child = nullptr;
  }

  __host__ ~ops_chain ()
  {
    for (int i = 0; i < n_read_write; i++)
    { m_read_write[i] -> ~multi_level_index(); free(m_read_write[i]); }
    free(m_read_write);

    for (int i = 0; i < n_read_only; i++)
    { m_read_only[i] -> ~multi_level_index(); free(m_read_only[i]); }
    free(m_read_only);

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
    else
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

  __host__ void print (const int op_id = 0, const int indent = 0, const bool recurse = true)
  {
    for (int i = 0; i < indent; i++) { printf("  "); }

    if (child == nullptr) { printf("%d: ", op_id); }

    switch(op_type)
    {
      case nop: printf("NOP "); break;
      case getrf: printf("GETRF "); break;
      case gessm: printf("GESSM "); break;
      case tstrf: printf("TSTRF "); break;
      case ssssm: printf("SSSSM "); break;
    }

    printf("%dRW: ", n_read_write);
    for (int i = 0; i < n_read_write; i++)
    { m_read_write[i] -> print_short(); printf(" "); }

    printf("%dR: ", n_read_only);
    for (int i = 0; i < n_read_only; i++)
    { m_read_only[i] -> print_short(); printf(" "); }

    printf("\n");

    if (child != nullptr && recurse) { child -> print(op_id, indent + 1, recurse); }

    int l_child = (child == nullptr) ? 1 : child -> length();
    if (next != nullptr && recurse) { next -> print(op_id + l_child, indent, recurse); }

    if ((next == nullptr && indent == 0) || !recurse) { printf("\n"); }
  }

};

#endif