
#ifndef _DEV_HIERARCHICAL_OPS_CUH
#define _DEV_HIERARCHICAL_OPS_CUH

#include <pspl.cuh>

class h_ops
{
private:

  operation_t op_type;
  h_index_linked_list *wr;
  h_index_linked_list *r;

public:

  __host__ h_ops (const operation_t op_in = nop)
  {
    op_type = op_in;
    wr = nullptr;
    r = nullptr;
  }

  __host__ ~h_ops ()
  {
    if (wr != nullptr) { delete wr; }
    if (r != nullptr) { delete r; }
  }

  __host__ void addWR(const h_index *in)
  {
    if (wr == nullptr) { wr = new h_index_linked_list(in); }
    else { wr -> hookup(in); }
  }

  __host__ void addR(const h_index *in)
  {
    if (r == nullptr) { r = new h_index_linked_list(in); }
    else { r -> hookup(in); }
  }

  __host__ int getN_WR() const
  {
    return (wr == nullptr) ? 0 : wr -> length();
  }

  __host__ int getN_R() const
  {
    return (r == nullptr) ? 0 : r -> length();
  }

  __host__ h_index * getI_WR(const int i) const
  {
    return (wr == nullptr) ? nullptr : wr -> lookup(i);
  }

  __host__ h_index * getI_R(const int i) const
  {
    return (r == nullptr) ? nullptr : r -> lookup(i);
  }

  __host__ void print() const
  {
    switch (op_type)
    {
    case nop: printf("NOP   "); break;
    case getrf: printf("GETRF "); break;
    case trsml: printf("TRSML "); break;
    case trsmr: printf("TRSMR "); break;
    case gemm: printf("GEMM  "); break;
    case pivot: printf("PIVOT "); break;
    }

    printf("%dx W: ", getN_WR());
    if (wr != nullptr) wr->print();

    printf("%dx R: ", getN_R());
    if (r != nullptr) r->print();

    printf("\n");
  }
  
};

class h_ops_tree 
{
private:

  h_ops *op;

  h_ops_tree * next;
  h_ops_tree * child;

public:

  __host__ h_ops_tree (const operation_t op_in = nop)
  {
    op = new h_ops(op_in);

    next = nullptr;
    child = nullptr;
  }

  __host__ ~h_ops_tree ()
  {
    delete op;
    if (child != nullptr) { delete child; }
    if (next != nullptr) { delete next; }
  }

  __host__ void addWR (const h_index *in)
  { op -> addWR(in); }

  __host__ void addR (const h_index *in)
  { op -> addR(in); }

  __host__ int getN_WR() const
  { return op -> getN_WR(); }

  __host__ int getN_R() const
  { return op -> getN_R(); }

  __host__ h_index * getI_WR (const int i) const
  { return op -> getI_WR(i); }

  __host__ h_index * getI_R (const int i) const
  { return op -> getI_R(i); }

  __host__ void hookup_next (h_ops_tree *chain)
  {
    if (next != nullptr)
    { next -> hookup_next (chain); }
    else
    { next = chain; }
  }

  __host__ void hookup_child (h_ops_tree *chain)
  {
    child = chain;
  }

  __host__ const h_ops_tree * lookup (const int index) const
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

    op -> print();

    if (child != nullptr) { child -> print(op_id, indent + 1); }

    if (next != nullptr) 
    {
      const int l_this = (child == nullptr) ? 1 : child -> length();
      next -> print(op_id + l_this, indent); 
    }
  }

};

#endif