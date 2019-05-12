
#ifndef _DEV_HIERARCHICAL_OPS_CUH
#define _DEV_HIERARCHICAL_OPS_CUH

#include <pspl.cuh>

class h_ops
{
protected:

  operation_t op_type;
  h_index * read_and_write;
  h_index * read_only;

public:

  __host__ h_ops ()
  {
    op_type = nop;
    read_and_write = nullptr;
    read_only = nullptr;
  }

  __host__ h_ops (const operation_t op_in, const h_index * M)
  {
    if (op_in != getrf_d) 
    { printf("Operation argument unmatched.\n"); }
    op_type = op_in;

    read_and_write = new h_index[1]{};
    M -> clone(&read_and_write[0]);

    read_only = nullptr;
  }

  __host__ h_ops (const operation_t op_in, const h_index * M1, const h_index * M2)
  {
    if (op_in < trsml_d || op_in > pivot_lr)
    { printf("Operation argument unmatched.\n"); }
    op_type = op_in;

    read_and_write = new h_index[1]{};
    M1 -> clone(&read_and_write[0]);

    read_only = new h_index[1]{};
    M2 -> clone(&read_only[0]);
  }

  __host__ h_ops (const operation_t op_in, const h_index * M1, const h_index * M2, const h_index * M3)
  {
    if (op_in < gemm_d_d_d || op_in > gemm_lr_lr_lr) 
    { printf("Operation argument unmatched.\n"); }
    op_type = op_in;

    read_and_write = new h_index[1]{};
    M1 -> clone(&read_and_write[0]);

    read_only = new h_index[2]{};
    M2 -> clone(&read_only[0]);
    M3 -> clone(&read_only[1]);
  }

  __host__ ~h_ops ()
  {
    delete[] read_and_write;
    delete[] read_only;
  }

  __host__ inline operation_t opType() const 
  { return op_type; }

  __host__ dependency_t checkDependencyFrom (const h_ops * op_from) const
  {
    int wr_from = 0, r_from = 0, wr_to = 0, r_to = 0;

    if (op_from -> opType() >= gemm_d_d_d)
    { r_from++; }
    if (op_from -> opType() >= trsml_d)
    { r_from++; }
    if (op_from -> opType() >= getrf_d)
    { wr_from++; }

    if (opType() >= gemm_d_d_d)
    { r_to++; }
    if (opType() >= trsml_d)
    { r_to++; }
    if (opType() >= getrf_d)
    { wr_to++; }

    int dep = (int) no_dep;

    for (int i = 0; i < wr_from * (wr_to + r_to); i++)
    {
      const int to = i / wr_from, from = i - to * wr_from;

      if (to < wr_to)
      {
        relation_t relation = read_and_write[to].compare(&(op_from -> read_and_write)[from]);
        switch (relation)
        {
        case diff_mat: case same_mat_diff_branch: case same_node_no_overlap: 
          break;
        case same_branch_diff_node: case same_node_overlapped: case same_index:
          dep |= (int) output_dep;
        }
      }
      else
      {
        relation_t relation = read_only[to - wr_to].compare(&(op_from -> read_and_write)[from]);
        switch (relation)
        {
        case diff_mat: case same_mat_diff_branch: case same_node_no_overlap: 
          break;
        case same_branch_diff_node: case same_node_overlapped: case same_index:
          dep |= (int) flow_dep;
        }
      }
    }

    for (int i = 0; i < r_from * wr_to; i++)
    {
      const int to = i / r_from, from = i - to * r_from;
      relation_t relation = read_and_write[to].compare(&(op_from -> read_only)[from]);
      switch (relation)
      {
      case diff_mat: case same_mat_diff_branch: case same_node_no_overlap: 
        break;
      case same_branch_diff_node: case same_node_overlapped: case same_index:
        dep |= (int) anti_dep;
      }
    }

    return (dependency_t)dep;
  }

  __host__ dependency_t checkDependencyTo (const h_ops * op_to) const
  {
    return op_to -> checkDependencyFrom(this);
  }

  __host__ int writeParametersTo (int * inst) const
  {
    int l_dims = 0, l_lds = 0, l_ts = 0;

    return l_dims + l_lds + l_ts;
  }

  __host__ unsigned long long int getFops () const
  {
    unsigned long long int accum = 0;

    return accum;
  }

  __host__ void print() const
  {
    if (opType() >= gemm_d_d_d)
    { printf("GEMM "); read_and_write[0].print(); read_only[0].print(); read_only[1].print(); printf("\n"); }
    else if (opType() >= trsml_d && opType() <= trsmr_lr)
    { printf("TRSM "); read_and_write[0].print(); read_only[0].print(); printf("\n"); }
    else if (opType() == getrf_d)
    { printf("GETRF "); read_and_write[0].print(); printf("\n"); }
    else
    { printf("NOP\n"); }
  }
  
};

class h_ops_tree : public h_ops
{
private:
  int l_children;
  h_ops_tree * children;

public:

  __host__ h_ops_tree () : h_ops ()
  { l_children = 0; children = nullptr; }

  __host__ h_ops_tree (const operation_t op_in, const h_index * M) : h_ops (op_in, M)
  { l_children = 0; children = nullptr; }

  __host__ h_ops_tree (const operation_t op_in, const h_index * M1, const h_index * M2) : h_ops (op_in, M1, M2)
  { l_children = 0; children = nullptr; }

  __host__ h_ops_tree (const operation_t op_in, const h_index * M1, const h_index * M2, const h_index * M3) : h_ops (op_in, M1, M2, M3)
  { l_children = 0; children = nullptr; }

  __host__ ~h_ops_tree ()
  { 
    if (l_children > 0) 
    { delete[] children; } 
  }

  __host__ h_ops_tree * getChild (const int index) const
  { return (index >= l_children) ? nullptr : &children[index]; }

  __host__ void setChild (h_ops_tree * op, const int index = -1)
  {
    if (index >= 0)
    { 
      if (index >= l_children) 
      { resizeChildren(index + 1); } 
      op -> clone(&children[index], true); 
    }
    else if (index == -1)
    { 
      resizeChildren(l_children + 1); 
      op -> clone(&children[l_children - 1], true); 
    }

  }

  __host__ void resizeChildren (const int length_in) 
  {
    if (length_in > 0 && length_in != l_children)
    { 
      h_ops_tree * neo = new h_ops_tree [length_in]{};

      for (int i = 0; i < l_children && i < length_in; i++)
      { children[i].clone(&neo[i], true); }

      if (l_children > 0) 
      { delete[] children; }

      children = neo;
      l_children = length_in;
    }
  }

  __host__ int length () const
  {
    if (this == nullptr)
    { return 0; }
    else if (l_children == 0) 
    { return 1; }
    else
    {
      int length = 0;
#pragma omp parallel for num_threads(2) reduction (+:length) 
      for (int i = 0; i < l_children; i++) 
      { length += children[i].length(); }
      return length;
    }
  }

  __host__ h_ops_tree * clone (h_ops_tree * addr = nullptr, const bool clone_child = false) const
  {
    if (this == nullptr)
    { return nullptr; }
    else if (addr == nullptr)
    { h_ops_tree * op = new h_ops_tree(); clone(op); return op; }
    else
    {
      if (opType() >= gemm_d_d_d)
      { 
        addr -> op_type = op_type;
        addr -> read_and_write = new h_index[1];
        read_and_write[0].clone(&(addr -> read_and_write)[0]);

        addr -> read_only = new h_index[2];
        read_only[0].clone(&(addr -> read_only)[0]);
        read_only[1].clone(&(addr -> read_only)[1]);
      }
      else if (opType() >= trsml_d && opType() <= pivot_lr)
      {         
        addr -> op_type = op_type;
        addr -> read_and_write = new h_index[1];
        read_and_write[0].clone(&(addr -> read_and_write)[0]);

        addr -> read_only = new h_index[1];
        read_only[0].clone(&(addr -> read_only)[0]);
      }
      else if (opType() == getrf_d)
      {         
        addr -> op_type = op_type;
        addr -> read_and_write = new h_index[1];
        read_and_write[0].clone(&(addr -> read_and_write)[0]);

        addr -> read_only = nullptr;
      }
      else
      {
        addr -> op_type = op_type;
        addr -> read_and_write = nullptr;
        addr -> read_only = nullptr;
      }

      if (clone_child)
      {
        addr -> l_children = l_children;
        addr -> children = (l_children > 0) ? new h_ops_tree [l_children] : nullptr;
        for (int i = 0; i < l_children; i++)
        { children[i].clone(&(addr -> children)[i], clone_child); }
      }
      else
      {
        addr -> l_children = 0;
        addr -> children = nullptr;
      }

      return addr;
    }
  }

  __host__ h_ops_tree * flatten (h_ops_tree * list = nullptr, const int list_index = 0) const
  {
    if (list == nullptr)
    {
      list = clone(nullptr, false);
      list -> resizeChildren(length());
    }

    int * work_index = new int[l_children];
    work_index[0] = list_index;

    for (int i = 1; i < l_children; i++)
    { work_index[i] = work_index[i - 1] + children[i - 1].length(); }

#pragma omp parallel for num_threads(2)
    for (int i = 0; i < l_children; i++)
    {
      if (children[i].l_children == 0)
      { children[i].clone(&(list -> children)[work_index[i]], false); }
      else
      { children[i].flatten (list, work_index[i]); }
    }

    delete[] work_index;
    return list;
  }

  __host__ unsigned long long int getFops() const
  {
    if (this == nullptr)
    { return 0; }
    else if (l_children == 0)
    { return h_ops::getFops(); }
    else
    { 
      unsigned long long int accum = 0;
      for (int i = 0; i < l_children; i++) 
      { accum += children[i].getFops(); }
      return accum;
    }
  }

  __host__ void print (const int op_id = 0, const int indent = 0) const
  {
    for (int i = 0; i < indent; i++) { printf("  "); }

    if (l_children == 0) { printf("%d: ", op_id); }

    h_ops::print();

    int offset = 0, l = length();
    for (int i = 0; i < l_children && offset < l; offset += children[i].length(), i++)
    { children[i].print(op_id + offset, indent + 1); }
  }

};

#endif