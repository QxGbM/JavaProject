
#ifndef _DEV_HIERARCHICAL_OPS_CUH
#define _DEV_HIERARCHICAL_OPS_CUH

#include <pspl.cuh>

class h_ops
{
protected:

  operation_t op_type;

  int n_rw;
  h_index * read_and_write;

  int n_ro;
  h_index * read_only;

public:

  __host__ h_ops ()
  {
    op_type = nop;
    n_rw = 0;
    read_and_write = nullptr;
    n_ro = 0;
    read_only = nullptr;
  }

  __host__ h_ops (const operation_t op_in, const h_index * M)
  {
    if (op_in != getrf)
    { printf("Operation argument unmatched.\n"); }
    op_type = op_in;

    read_and_write = new h_index[1]{};
    M -> clone(&read_and_write[0]);
    n_rw = 1;

    read_only = nullptr;
    n_ro = 0;
  }

  __host__ h_ops (const operation_t op_in, const h_index * M1, const h_index * M2)
  {
    if (op_in < trsml || op_in > pivot)
    { printf("Operation argument unmatched.\n"); }
    op_type = op_in;

    read_and_write = new h_index[1]{};
    M1 -> clone(&read_and_write[0]);
    n_rw = 1;

    read_only = new h_index[1]{};
    M2 -> clone(&read_only[0]);
    n_ro = 1;
  }

  __host__ h_ops (const operation_t op_in, const h_index * M1, const h_index * M2, const h_index * M3)
  {
    if (op_in != gemm) 
    { printf("Operation argument unmatched.\n"); }
    op_type = op_in;

    read_and_write = new h_index[1]{};
    M1 -> clone(&read_and_write[0]);
    n_rw = 1;

    read_only = new h_index[2]{};
    M2 -> clone(&read_only[0]);
    M3 -> clone(&read_only[1]);
    n_ro = 2;
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
    int rw_from = op_from -> n_rw, ro_from = op_from -> n_ro, rw_to = n_rw, ro_to = n_ro, dep = (int) no_dep;

    for (int i = 0; i < rw_from * (rw_to + ro_to); i++)
    {
      const int to = i / rw_from, from = i - to * rw_from;

      if (to < rw_to)
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
        relation_t relation = read_only[to - rw_to].compare(&(op_from -> read_and_write)[from]);
        switch (relation)
        {
        case diff_mat: case same_mat_diff_branch: case same_node_no_overlap: 
          break;
        case same_branch_diff_node: case same_node_overlapped: case same_index:
          dep |= (int) flow_dep;
        }
      }
    }

    for (int i = 0; i < ro_from * rw_to; i++)
    {
      const int to = i / ro_from, from = i - to * ro_from;
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
  { return op_to -> checkDependencyFrom(this); }

  __host__ int getDataPointers (void ** data_ptrs, void ** tmp_ptrs) const
  {
    int accum = 0;
    for (int i = 0; i < n_rw; i++)
    { accum += read_and_write[i].getDataPointers (&data_ptrs[accum], tmp_ptrs); }

    for (int i = 0; i < n_ro; i++)
    { accum += read_only[i].getDataPointers (&data_ptrs[accum], tmp_ptrs); }
    
    return accum;
  }

  __host__ int writeOpParametersTo (int * inst, const int * mapping) const
  {

    switch (opType())
    {
    case getrf:
    {
      int M, offset_m, nx, ny, ld;

      if (read_and_write[0].isDense())
      {
        M = mapping[0];
        offset_m = read_and_write[0].getOffset_x();
        nx = read_and_write[0].getNx();
        ny = read_and_write[0].getNy();
        ld = read_and_write[0].getLd_x();
      }
      else
      { printf("Error: GETRF on incompatible block.\n"); return 0; }

      inst[0] = (int) getrf;
      inst[1] = M;
      inst[2] = offset_m;
      inst[3] = nx; 
      inst[4] = ny; 
      inst[5] = ld;
      return 6;
    }
    case trsml:
    {
      int B, L, offset_b, offset_l, nx_b, ny_b, nx_l, ld_b, ld_l, b_T;

      if (read_and_write[0].isU() && read_only[0].isDense())
      {
        B = mapping[0];
        L = mapping[2];
        offset_b = read_and_write[0].getOffset_y();
        offset_l = read_only[0].getOffset_x();
        nx_b = read_and_write[0].getRank();
        ny_b = read_and_write[0].getNy();
        ny_b = ny_b > read_only[0].getNy() ? read_only[0].getNy() : ny_b;
        nx_l = read_only[0].getNx();
        ld_b = read_and_write[0].getLd_y();
        ld_l = read_only[0].getLd_x();
        b_T = read_and_write[0].getTranspose();
      }
      else if (read_and_write[0].isDense() && read_only[0].isDense())
      {
        B = mapping[0];
        L = mapping[1];
        offset_b = read_and_write[0].getOffset_x();
        offset_l = read_only[0].getOffset_x();
        nx_b = read_and_write[0].getNx();
        ny_b = read_and_write[0].getNy();
        ny_b = ny_b > read_only[0].getNy() ? read_only[0].getNy() : ny_b;
        nx_l = read_only[0].getNx();
        ld_b = read_and_write[0].getLd_x();
        ld_l = read_only[0].getLd_x();
        b_T = read_and_write[0].getTranspose();
      }
      else
      { printf("Error: TRSML on incompatible block.\n"); return 0; }

      inst[0] = (int) trsml;
      inst[1] = B;
      inst[2] = L;
      inst[3] = offset_b;
      inst[4] = offset_l;
      inst[5] = nx_b;
      inst[6] = ny_b;
      inst[7] = nx_l;
      inst[8] = ld_b;
      inst[9] = ld_l;
      inst[10] = b_T;
      return 11;
    }
    case trsmr:
    {
      int B, U, offset_b, offset_u, nx_b, ny_b, ny_u, ld_b, ld_u, b_T;

      if (read_and_write[0].isVT() && read_only[0].isDense())
      {
        B = mapping[0];
        U = mapping[2];
        offset_b = read_and_write[0].getOffset_x();
        offset_u = read_only[0].getOffset_x();
        nx_b = read_and_write[0].getNx();
        nx_b = nx_b > read_only[0].getNx() ? read_only[0].getNx() : nx_b;
        ny_b = read_and_write[0].getRank();
        ny_u = read_only[0].getNy();
        ld_b = read_and_write[0].getLd_x();
        ld_u = read_only[0].getLd_x();
        b_T = read_and_write[0].getTranspose();
      }
      else if (read_and_write[0].isDense() && read_only[0].isDense())
      {
        B = mapping[0];
        U = mapping[1];
        offset_b = read_and_write[0].getOffset_x();
        offset_u = read_only[0].getOffset_x();
        nx_b = read_and_write[0].getNx();
        nx_b = nx_b > read_only[0].getNx() ? read_only[0].getNx() : nx_b;
        ny_b = read_and_write[0].getNy();
        ny_u = read_only[0].getNy();
        ld_b = read_and_write[0].getLd_x();
        ld_u = read_only[0].getLd_x();
        b_T = read_and_write[0].getTranspose();
      }
      else
      { printf("Error: TRSMR on incompatible block.\n"); return 0; }

      inst[1] = B;
      inst[2] = U;
      inst[3] = offset_b;
      inst[4] = offset_u;
      inst[5] = nx_b;
      inst[6] = ny_b;
      inst[7] = ny_u;
      inst[8] = ld_b;
      inst[9] = ld_u;
      inst[10] = b_T;
      return 11;
    }
    case gemm:
    {
      /*int M, A, B, C, D, offset_m, offset_a, offset_b, offset_c, offset_d, m, n, k, l, o, ld_m, ld_a, ld_b, ld_c, ld_d, a_T, b_T, c_T, d_T;

      if (read_and_write[0].isU() && read_only[0].isU())
      {
        if (read_only[0].isDense())
        {

          m = read_and_write[0].getNy();
          n = read_and_write[0].getNx();
          k = read_only[0].getNx();
          m = m > read_only[0].getNy() ? read_only[0].getNy() : m;
          n = n > read_only[1].getNx() ? read_only[1].getNx() : n;
          k = k > read_only[1].getNy() ? read_only[1].getNy() : k;
        }
      }


      inst[0] = m; inst[1] = n; inst[2] = k;*/
      return 0;
    }
    default:
    { return 0; }
    }


  }

  __host__ unsigned long long int getFops () const
  {
    unsigned long long int accum = 0;

    return accum;
  }

  __host__ void print() const
  {
    if (opType() == gemm)
    { printf("GEMM "); read_and_write[0].print(); read_only[0].print(); read_only[1].print(); printf("\n"); }
    else if (opType() == pivot)
    { printf("PVT "); read_and_write[0].print(); read_only[0].print(); printf("\n"); }
    else if (opType() == accum)
    { printf("ACCM "); read_and_write[0].print(); read_only[0].print(); printf("\n"); }
    else if (opType() == trsmr)
    { printf("TRSMR "); read_and_write[0].print(); read_only[0].print(); printf("\n"); }
    else if (opType() == trsml)
    { printf("TRSML "); read_and_write[0].print(); read_only[0].print(); printf("\n"); }
    else if (opType() == getrf)
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
#pragma omp parallel for reduction (+:length) if (omp_in_parallel() == 0)
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
      if (opType() == gemm)
      { 
        addr -> op_type = op_type;
        addr -> read_and_write = new h_index[1];
        read_and_write[0].clone(&(addr -> read_and_write)[0]);
        addr -> n_rw = 1;

        addr -> read_only = new h_index[2];
        read_only[0].clone(&(addr -> read_only)[0]);
        read_only[1].clone(&(addr -> read_only)[1]);
        addr -> n_ro = 2;
      }
      else if (opType() >= trsml && opType() <= pivot)
      {         
        addr -> op_type = op_type;
        addr -> read_and_write = new h_index[1];
        read_and_write[0].clone(&(addr -> read_and_write)[0]);
        addr -> n_rw = 1;

        addr -> read_only = new h_index[1];
        read_only[0].clone(&(addr -> read_only)[0]);
        addr -> n_ro = 1;
      }
      else if (opType() == getrf)
      {         
        addr -> op_type = op_type;
        addr -> read_and_write = new h_index[1];
        read_and_write[0].clone(&(addr -> read_and_write)[0]);
        addr -> n_rw = 1;

        addr -> read_only = nullptr;
        addr -> n_ro = 0;
      }
      else
      {
        addr -> op_type = op_type;
        addr -> read_and_write = nullptr;
        addr -> n_rw = 0;
        addr -> read_only = nullptr;
        addr -> n_ro = 0;
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

#pragma omp parallel for if (omp_in_parallel() == 0)
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