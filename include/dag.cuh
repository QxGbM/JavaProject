#ifndef _DAG_CUH
#define _DAG_CUH

#include <dev_hierarchical.cuh>

enum matrix_op_t {
  nop,
  getrf,
  getrf_pivot,
  apply_pivot,
  trsm,
  gemm,
};

__host__ int calc_load (int op)
{
  int load_table[] = {1, 1, 1, 1, 1, 1};
  return load_table[op];
}

struct dag {

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

    __host__ int length ()
    {
      int l_child = (child == nullptr) ? 0 : child -> length();
      int l_next = (next == nullptr) ? 0 : next -> length();
      return 1 + l_child + l_next;
    }

    __host__ void print (const bool indent = true)
    {
      for (int i = 0; i < (dest -> levels) && indent; i++) { printf("  "); }

      switch(op_type)
      {
        case nop: printf("NOP "); break;
        case getrf: printf("GETRF "); break;
        case getrf_pivot: printf("GETRFP "); break;
        case apply_pivot: printf("PIVOT "); break;
        case trsm: printf("TRSM "); break;
        case gemm: printf("GEMM "); break;
      }

      if (dest != nullptr) { dest -> print_short(); printf(", "); }
      else { printf("_ "); }
      if (m1 != nullptr) { m1 -> print_short(); printf(", "); }
      else { printf("_ "); }
      if (m2 != nullptr) { m2 -> print_short(); }
      else { printf("_"); }
      printf("\n");
    }

    __host__ struct ops_chain * getops_hgetrf (const struct dev_hierarchical <double> *a)
    {
      struct ops_chain *head, *acc;
      int n = (a -> nx) * (a -> ny);
      for(int i = 0; i < n; i++) {}

      return acc;
    }

  };

  struct ops_chain *ops;

  __host__ dag () {
    ops = nullptr;
  }
  

};


#endif