
#ifndef _OPS_CUH
#define _OPS_CUH

#include <pspl.cuh>

class ops_chain 
{
private:

  class index_chain
  {
  private:
    multi_level_index *index;
    index_chain *next;

  public:
    __host__ index_chain (const multi_level_index * in) 
    {
      index = in -> clone();
      next = nullptr;
    }

    __host__ ~index_chain()
    {
      delete index;
      if (next != nullptr) { delete next; }
    }

    __host__ void hookup (const multi_level_index * in)
    {
      if (next == nullptr) { next = new index_chain(in); }
      else { next -> hookup(in); }
    }

    __host__ int length() const
    {
      return 1 + ((next == nullptr) ? 0 : next -> length());
    }

    __host__ multi_level_index * lookup(const int i) const
    {
      if (i <= 0) { return index; }
      else { return (next == nullptr) ? nullptr : next -> lookup(i - 1); }
    }

    __host__ void print() const
    {
      index -> print_short();
      printf(" ");
      if (next != nullptr) { next -> print(); }
    }

  };

  matrix_op_t op_type;

  index_chain *wr;
  index_chain *r;

  ops_chain *next;
  ops_chain *child;

public:

  __host__ ops_chain (const matrix_op_t opin = nop)
  {
    op_type = opin;

    wr = nullptr;
    r = nullptr;
    next = nullptr;
    child = nullptr;
  }

  __host__ ~ops_chain ()
  {
    if (wr != nullptr) { delete wr; }
    if (r != nullptr) { delete r; }

    if (child != nullptr) { delete child; }
    if (next != nullptr) { delete next; }
  }

  __host__ void addWR (const multi_level_index *in)
  {
    if (wr == nullptr) { wr = new index_chain(in); }
    else { wr -> hookup(in); }
  }

  __host__ void addR (const multi_level_index *in)
  {
    if (r == nullptr) { r = new index_chain(in); }
    else { r -> hookup(in); }
  }

  __host__ int getN_WR() const
  { return (wr == nullptr) ? 0 : wr -> length(); }

  __host__ int getN_R() const
  { return (r == nullptr) ? 0 : r -> length(); }

  __host__ multi_level_index * getI_WR (const int i) const
  { return (wr == nullptr) ? nullptr : wr -> lookup(i); }

  __host__ multi_level_index * getI_R (const int i) const
  { return (r == nullptr) ? nullptr : r -> lookup(i); }

  __host__ void hookup_next (ops_chain *chain)
  {
    if (next != nullptr)
    { next -> hookup_next (chain); }
    else
    { next = chain; }
  }

  __host__ void hookup_child (ops_chain *chain)
  {
    child = chain;
  }

  __host__ const ops_chain * lookup (const int index) const
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

  __host__ int length () const 
  {
    int l_child = (child == nullptr) ? 1 : child -> length();
    int l_next = (next == nullptr) ? 0 : next -> length();
    return l_child + l_next;
  }

  __host__ void print (const int op_id = 0, const int indent = 0) const
  {
    for (int i = 0; i < indent; i++) { printf("  "); }

    if (child == nullptr) { printf("%d: ", op_id); }

    switch(op_type)
    {
      case nop: printf("NOP   "); break;
      case getrf: printf("GETRF "); break;
      case trsml: printf("TRSML "); break;
      case trsmr: printf("TRSMR "); break;
      case gemm: printf("GEMM  "); break;
      case pivot: printf("PIVOT "); break;
    }

    printf("%dx W: ", getN_WR());
    if (wr != nullptr) wr -> print();

    printf("%dx R: ", getN_R());
    if (r != nullptr) r -> print();

    printf("\n");

    if (child != nullptr) { child -> print(op_id, indent + 1); }

    if (next != nullptr) 
    {
      const int l_this = (child == nullptr) ? 1 : child -> length();
      next -> print(op_id + l_this, indent); 
    }
  }

};

#endif