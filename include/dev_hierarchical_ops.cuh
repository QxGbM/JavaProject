
#ifndef _DEV_HIERARCHICAL_OPS_CUH
#define _DEV_HIERARCHICAL_OPS_CUH

#include <pspl.cuh>

class h_ops
{
private:

  operation_t op_type;
  h_index *wr;
  h_index *r;
  int *dims;
  int *lds;

public:

  __host__ h_ops (const operation_t op_in = nop)
  {
    op_type = op_in;
    wr = new h_index[0];
    r = new h_index[0];
    dims = new int[0];
    lds = new int[0];
  }

  __host__ h_ops (const operation_t op_in, const h_index * M, const int nx, const int ny, const int ld)
  {
    op_type = op_in;
    wr = new h_index[1]{ *(M -> clone()) };
    r = new h_index[0];
    dims = new int[2]{ nx, ny };
    lds = new int[1]{ ld };
  }

  __host__ h_ops (const operation_t op_in, const h_index * B, const h_index * M, const int nx_b, const int ny_b, const int dim_m, const int ld_b, const int ld_m)
  {
    op_type = op_in;
    wr = new h_index[1]{ *(B -> clone()) };
    r = new h_index[1]{ *(M -> clone()) };
    dims = new int[3]{ nx_b, ny_b, dim_m };
    lds = new int[2]{ ld_b, ld_m };
  }

  __host__ h_ops (const operation_t op_in, const h_index * M, const h_index * A, const h_index * B, const int m, const int n, const int k, const int ld_m, const int ld_a, const int ld_b)
  {
    op_type = op_in;
    wr = new h_index[1]{ *(M -> clone()) };
    r = new h_index[2]{ *(A -> clone()), *(B -> clone()) };
    dims = new int[3]{ m, n, k };
    lds = new int[3]{ ld_m, ld_a, ld_b };
  }

  __host__ ~h_ops ()
  {
    delete[] r;
    delete[] wr;
    delete[] dims;
    delete[] lds;
  }

  __host__ dependency_t checkDependencyFrom (const h_ops * op_from) const
  {
    return no_dep;
  }

  __host__ dependency_t checkDependencyTo (const h_ops * op_to) const
  {
    return no_dep;
  }

  __host__ void print() const
  {
    switch (op_type)
    {
    case nop: 
      printf("NOP "); 
      break;
    case getrf: 
      printf("GETRF "); 
      wr[0].printShort(); printf(" (%d x %d by %d) ", dims[1], dims[0], lds[0]); 
      break;
    case trsml: 
      printf("TRSML "); 
      wr[0].printShort(); printf(" (%d x %d by %d) ", dims[1], dims[0], lds[0]); 
      r[0].printShort(); printf(" (%d x %d by %d) ", dims[1], dims[2], lds[1]); 
      break;
    case trsmr: 
      printf("TRSMR "); 
      wr[0].printShort(); printf(" (%d x %d by %d) ", dims[1], dims[0], lds[0]);
      r[0].printShort(); printf(" (%d x %d by %d) ", dims[2], dims[0], lds[1]);
      break;
    case gemm: 
      printf("GEMM  "); 
      wr[0].printShort(); printf(" (%d x %d by %d) ", dims[0], dims[1], lds[0]);
      r[0].printShort(); printf(" (%d x %d by %d) ", dims[0], dims[2], lds[1]);
      r[1].printShort(); printf(" (%d x %d by %d) ", dims[2], dims[1], lds[2]);
      break;
    case pivot: 
      printf("PIVOT "); 
      wr[0].printShort(); printf(" (%d x %d by %d) ", dims[1], dims[0], lds[0]); 
      break;
    }

    printf("\n");
  }
  
};

class h_ops_tree 
{
private:

  h_ops op;

  h_ops_tree * next;
  h_ops_tree * child;

public:

  __host__ h_ops_tree (const h_ops * op_in)
  {
    op = * op_in;

    next = nullptr;
    child = nullptr;
  }

  __host__ ~h_ops_tree ()
  {
    delete child;
    delete next;
  }

  __host__ void hookup_next (h_ops_tree *tree)
  {
    if (next != nullptr)
    { next -> hookup_next (tree); }
    else
    { next = tree; }
  }

  __host__ void hookup_next (h_ops *op)
  {
    hookup_next(new h_ops_tree(op));
  }

  __host__ void hookup_child (h_ops_tree *tree)
  {
    if (child != nullptr)
    { child -> hookup_next (tree); }
    else
    { child = tree; }
  }

  __host__ void hookup_child (h_ops *op)
  {
    hookup_child(new h_ops_tree(op));
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

    op.print();

    if (child != nullptr) { child -> print(op_id, indent + 1); }

    if (next != nullptr) 
    {
      const int l_this = (child == nullptr) ? 1 : child -> length();
      next -> print(op_id + l_this, indent); 
    }
  }

};

#endif