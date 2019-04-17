
#ifndef _DEV_HIERARCHICAL_OPS_CUH
#define _DEV_HIERARCHICAL_OPS_CUH

#include <pspl.cuh>

class h_ops
{
protected:

  operation_t op_type;
  h_index *wr;
  h_index *r;
  int *dims;
  int *lds;
  int *ts;

public:

  __host__ h_ops (const operation_t op_in = nop)
  {
    if (op_in != nop) { printf("Operation argument unmatched.\n"); }
    op_type = op_in;

    wr = new h_index[0]{};
    r = new h_index[0]{};

    dims = new int[0];
    lds = new int[0];
    ts = new int[0];
  }

  __host__ h_ops (const operation_t op_in, const h_index * M, const int nx, const int ny, const int ld)
  {
    if (op_in != getrf) { printf("Operation argument unmatched.\n"); }
    op_type = op_in;

    wr = new h_index[1]{};
    M -> cloneTo(&wr[0]);
    r = new h_index[0]{};

    dims = new int[2]{ nx, ny };
    lds = new int[1]{ ld };
    ts = new int[0]{};
  }

  __host__ h_ops (const operation_t op_in, const h_index * B, const h_index * M, const int nx_b, const int ny_b, const int dim_m, 
    const int ld_b, const int ld_m)
  {
    if (op_in != trsml && op_in != trsmr) { printf("Operation argument unmatched.\n"); }
    op_type = op_in;

    wr = new h_index[1]{};
    B -> cloneTo(&wr[0]);
    r = new h_index[1]{};
    M -> cloneTo(&r[0]);

    dims = new int[3]{ nx_b, ny_b, dim_m };
    lds = new int[2]{ ld_b, ld_m };
    ts = new int[0]{};
  }

  __host__ h_ops (const operation_t op_in, const h_index * M, const h_index * A, const h_index * B, const int m, const int n, const int k, 
    const int ld_m, const int ld_a, const int ld_b, const bool A_T, const bool B_T)
  {
    if (op_in != gemm) { printf("Operation argument unmatched.\n"); }
    op_type = op_in;

    wr = new h_index[1]{};
    M -> cloneTo(&wr[0]);
    r = new h_index[2]{};
    A -> cloneTo(&r[0]);
    B -> cloneTo(&r[1]);

    dims = new int[3]{ m, n, k };
    lds = new int[3]{ ld_m, ld_a, ld_b };
    ts = new int[2]{ (int) A_T, (int) B_T };
  }

  __host__ h_ops(const operation_t op_in, const h_index * M, const h_index * P, const int nx, const int ny, const int ld, const bool p_T)
  {
    if (op_in != pivot) { printf("Operation argument unmatched.\n"); }
    op_type = op_in;

    wr = new h_index[1]{};
    M -> cloneTo(&wr[0]);
    r = new h_index[1]{};
    P -> cloneTo(&r[0]);

    dims = new int[2]{ nx, ny };
    lds = new int[1]{ ld };
    ts = new int[1]{ (int) p_T };
  }

  __host__ h_ops (const operation_t op_in, const h_index * LR, const h_index * M, const int nx_lr, const int ny_lr, const int dim_m, 
    const int ld_lr, const int ld_m, const bool lr_T)
  {
    if (op_in != trsml_lr && op_in != trsmr_lr) { printf("Operation argument unmatched.\n"); }
    op_type = op_in;

    wr = new h_index[1]{};
    LR -> cloneTo(&wr[0]);
    r = new h_index[1]{};
    M -> cloneTo(&r[0]);

    dims = new int[3]{ nx_lr, ny_lr, dim_m };
    lds = new int[2]{ ld_lr, ld_m };
    ts = new int[1]{ (int) lr_T };
  }

  __host__ h_ops (const operation_t op_in, const h_index * M, const h_index * A, const h_index * B, const h_index * C,
    const int m, const int n, const int k, const int l, const int ld_m, const int ld_a, const int ld_b, const int ld_c,
    const bool a_T, const bool b_T, const bool c_T)
  {
    if (op_in != gemm3) { printf("Operation argument unmatched.\n"); }
    op_type = op_in;

    wr = new h_index[1]{};
    M -> cloneTo(&wr[0]);
    r = new h_index[3]{};
    A -> cloneTo(&r[0]);
    B -> cloneTo(&r[1]);
    C -> cloneTo(&r[2]);

    dims = new int[4]{ m, n, k, l };
    lds = new int[4]{ ld_m, ld_a, ld_b, ld_c };
    ts = new int[3]{ (int) a_T, (int) b_T, (int) c_T };
  }

  __host__ h_ops (const operation_t op_in, const h_index * M, const h_index * A, const h_index * B, const h_index * C, const h_index * D,
    const int m, const int n, const int k, const int l, const int o, const int ld_m, const int ld_a, const int ld_b, const int ld_c, const int ld_d,
    const bool a_T, const bool b_T, const bool c_T, const bool d_T)
  {
    if (op_in != gemm4) { printf("Operation argument unmatched.\n"); }
    op_type = op_in;

    wr = new h_index[1]{};
    M -> cloneTo(&wr[0]);
    r = new h_index[4]{};
    A -> cloneTo(&r[0]);
    B -> cloneTo(&r[1]);
    C -> cloneTo(&r[2]);
    D -> cloneTo(&r[3]);

    dims = new int[5]{ m, n, k, l, o };
    lds = new int[5]{ ld_m, ld_a, ld_b, ld_c, ld_d };
    ts = new int[4]{ (int) a_T, (int) b_T, (int) c_T, (int) d_T };
  }

  __host__ h_ops (const operation_t op_in, const h_index * M, const h_index * A, const h_index * B, const h_index * C, const h_index * D, const h_index * E, 
    const int m, const int n, const int k, const int l, const int o, const int p,
    const int ld_m, const int ld_a, const int ld_b, const int ld_c, const int ld_d, const int ld_e,
    const bool a_T, const bool b_T, const bool c_T, const bool d_T, const bool e_T)
  {
    if (op_in != gemm5) { printf("Operation argument unmatched.\n"); }
    op_type = op_in;

    wr = new h_index[1]{};
    M -> cloneTo(&wr[0]);
    r = new h_index[5]{};
    A -> cloneTo(&r[0]);
    B -> cloneTo(&r[1]);
    C -> cloneTo(&r[2]);
    D -> cloneTo(&r[3]);
    E -> cloneTo(&r[4]);

    dims = new int[6]{ m, n, k, l, o, p };
    lds = new int[6]{ ld_m, ld_a, ld_b, ld_c, ld_d, ld_e };
    ts = new int[5]{ (int) a_T, (int) b_T, (int) c_T, (int) d_T, (int) e_T };
  }

  __host__ ~h_ops ()
  {
    delete[] wr;
    delete[] r;

    delete[] dims;
    delete[] lds;
    delete[] ts;
  }

  __host__ operation_t opType() const { return op_type; }

  __host__ int wr_nx (const int i) const
  {
    switch (i)
    {
    case 0:
      switch (op_type) 
      {
      case nop: return 0; 
      case gemm: return dims[1]; 
      default: return dims[0];
      }
    default:
      return 0;
    }
  }

  __host__ int wr_ny (const int i) const
  {
    switch (i)
    {
    case 0:
      switch (op_type) 
      {
      case nop: return 0; 
      case gemm: return dims[0]; 
      default: return dims[1];
      }
    default:
      return 0;
    }
  }

  __host__ int wr_ld (const int i) const
  {
    switch (i)
    {
    case 0:
      switch (op_type)
      {
      case nop: return 0;
      default: return lds[0];
      }
    default:
      return 0;
    }
  }

  __host__ int wr_T (const int i) const 
  { 
    return 0; 
  }

  __host__ int r_nx (const int i) const
  {
    switch (i)
    {
    case 0:
      switch (op_type)
      {
      case gemm: case trsml: return dims[2];
      case trsmr: return dims[0];
      default: return 0;
      }
    case 1:
      switch (op_type)
      {
      case gemm: return dims[1];
      default: return 0;
      }
    default:
      return 0;
    }
  }

  __host__ int r_ny (const int i) const
  {
    switch (i)
    {
    case 0:
      switch (op_type)
      {
      case gemm: return dims[0];
      case trsml: return dims[1];
      case trsmr: return dims[2];
      default: return 0;
      }
    case 1:
      switch (op_type)
      {
      case gemm: return dims[2];
      default: return 0;
      }
    default:
      return 0;
    }
  }

  __host__ int r_ld (const int i) const
  {
    switch (i)
    {
    case 0:
      switch (op_type)
      {
      case trsml: case trsmr: case gemm: return lds[1];
      default: return 0;
      }
    case 1:
      switch (op_type)
      {
      case gemm: return lds[2];
      default: return 0;
      }
    default:
      return 0;
    }
  }

  __host__ int r_T (const int i) const
  {
    switch (i)
    {
    case 0:
      switch (op_type)
      {
      case gemm: return ts[1];
      default: return 0;
      }
    case 1:
      switch (op_type)
      {
      case gemm: return ts[2];
      default: return 0;
      }
    default:
      return 0;
    }
  }

  __host__ int l_wr () const
  {
    switch (op_type)
    {
    case nop: return 0;
    default: return 1;
    }
  }

  __host__ int l_r () const
  {
    switch (op_type)
    {
    case gemm: return 2;
    case trsml: case trsmr: case pivot: return 1;
    default: return 0;
    }
  }

  template <class T> __host__ T * wr_ptr (const int i, const dev_hierarchical <T> *h) const 
  {
    switch (i)
    {
    case 0:
      switch (op_type)
      {
      case nop: return nullptr;
      default: return h -> lookup(&wr[0]);
      }
    default:
      return nullptr;
    }
  }

  template <class T> __host__ int * wr_pivot_ptr (const int i, const dev_hierarchical <T> *h) const 
  {
    switch (i)
    {
    case 0:
      switch (op_type)
      {
      case getrf: return h -> lookup_pivot (&wr[0]);
      default: return nullptr;
      }
    default:
      return nullptr;
    }
  }

  template <class T> __host__ T * r_ptr (const int i, const dev_hierarchical <T> *h) const 
  {
    switch (i)
    {
    case 0:
      switch (op_type)
      {
      case trsml: case trsmr: case gemm: return h -> lookup(&r[0]);
      default: return nullptr;
      }
    case 1:
      switch (op_type)
      {
      case gemm: return h -> lookup(&r[1]);
      default: return nullptr;
      }
    default:
      return nullptr;
    }
  }

  template <class T> __host__ int * r_pivot_ptr (const int i, const dev_hierarchical <T> *h) const 
  {
    switch (i)
    {
    case 0:
      switch (op_type)
      {
      case pivot: return h -> lookup_pivot (&r[0]);
      default: return nullptr;
      }
    default:
      return nullptr;
    }
  }

  __host__ dependency_t checkDependencyFrom (const h_ops * op_from) const
  {
    int wr_from = 0, r_from = 0, wr_to = 0, r_to = 0;

    switch (op_from -> op_type)
    {
    case gemm: r_from++;
    case trsml: case trsmr: case pivot: r_from++;
    case getrf: wr_from++;
    case nop: break;
    }

    switch (op_type)
    {
    case gemm: r_to++;
    case trsml: case trsmr: case pivot: r_to++;
    case getrf: wr_to++;
    case nop: break;
    }

    dependency_t dep = no_dep;

    for (int i = 0; i < wr_from; i++)
    {
      for (int j = 0; j < r_to; j++)
      {
        relation_t relation = r[j].compare(r_nx(j), r_ny(j), r_ld(j), &(op_from -> wr)[i], op_from -> wr_nx(i), op_from -> wr_ny(i), op_from -> wr_ld(i));
        switch (relation)
        {
        case diff_matrix: case no_relation: case diff_offset_no_overlap: break;
        case diff_offset_overlapped: case same_index: case contains: case contained:
          dep = (dependency_t) ((int) dep | (int) flow_dep);
        }
      }

      for (int j = 0; j < wr_to; j++)
      {
        relation_t relation = wr[j].compare(wr_nx(j), wr_ny(j), wr_ld(j), &(op_from -> wr)[i], op_from -> wr_nx(i), op_from -> wr_ny(i), op_from -> wr_ld(i));
        switch (relation)
        {
        case diff_matrix: case no_relation: case diff_offset_no_overlap: break;
        case diff_offset_overlapped: case same_index: case contains: case contained:
          dep = (dependency_t) ((int) dep | (int) output_dep);
        }
      }
    }

    for (int i = 0; i < wr_to; i++)
    {
      for (int j = 0; j < r_from; j++)
      {
        relation_t relation = wr[j].compare(wr_nx(j), wr_ny(j), wr_ld(j), &(op_from -> r)[i], op_from -> r_nx(i), op_from -> r_ny(i), op_from -> r_ld(i));
        switch (relation)
        {
        case diff_matrix: case no_relation: case diff_offset_no_overlap: break;
        case diff_offset_overlapped: case same_index: case contains: case contained:
          dep = (dependency_t) ((int) dep | (int) anti_dep);
        }
      }
    }

    return dep;
  }

  __host__ dependency_t checkDependencyTo (const h_ops * op_to) const
  {
    return op_to -> checkDependencyFrom(this);
  }

  __host__ h_ops * clone() const
  {
    switch (op_type)
    {
    case nop:
      return new h_ops ();
    case getrf:
      return new h_ops (op_type, &wr[0], dims[0], dims[1], lds[0]);
    case trsml: case trsmr:
      return new h_ops (op_type, &wr[0], &r[0], dims[0], dims[1], dims[2], lds[0], lds[1]);
    case gemm:
      return new h_ops (op_type, &wr[0], &r[0], &r[1], dims[0], dims[1], dims[2], lds[0], lds[1], lds[2], (bool)ts[0], (bool)ts[1]);
    case pivot:
      return new h_ops (op_type, &wr[0], &r[0], dims[0], dims[1], lds[0], (bool)ts[0]);
    case trsml_lr: case trsmr_lr:
      return new h_ops (op_type, &wr[0], &r[0], dims[0], dims[1], dims[2], lds[0], lds[1], (bool)ts[0]);
    case gemm3:
      return new h_ops (op_type, &wr[0], &r[0], &r[1], &r[2], dims[0], dims[1], dims[2], dims[3],
        lds[0], lds[1], lds[2], lds[3], (bool)ts[0], (bool)ts[1], (bool)ts[2]);
    case gemm4:
      return new h_ops (op_type, &wr[0], &r[0], &r[1], &r[2], &r[3], dims[0], dims[1], dims[2], dims[3], dims[4], 
        lds[0], lds[1], lds[2], lds[3], lds[4], (bool)ts[0], (bool)ts[1], (bool)ts[2], (bool)ts[3]);
    case gemm5:
      return new h_ops (op_type, &wr[0], &r[0], &r[1], &r[2], &r[3], &r[4], dims[0], dims[1], dims[2], dims[3], dims[4], dims[5],
        lds[0], lds[1], lds[2], lds[3], lds[4], lds[5], (bool)ts[0], (bool)ts[1], (bool)ts[2], (bool)ts[3], (bool)ts[4]);
    default:
      return nullptr;
    }
  }

  __host__ int writeParametersTo (int * inst) const
  {
    int l_dims = 0, l_lds = 0, l_ts = 0;

    switch (op_type)
    {
    case nop:
      break;
    case getrf:
      l_dims = 2; l_lds = 1; break;
    case trsml: case trsmr:
      l_dims = 3; l_lds = 2; break;
    case gemm:
      l_dims = 3; l_lds = 3; l_ts = 2; break;
    case pivot:
      l_dims = 2; l_lds = 1; l_ts = 1; break;
    case trsml_lr: case trsmr_lr:
      l_dims = 3; l_lds = 2; l_ts = 1; break;
    case gemm3:
      l_dims = 4; l_lds = 4; l_ts = 3; break;
    case gemm4:
      l_dims = 5; l_lds = 5; l_ts = 4; break;
    case gemm5:
      l_dims = 6; l_lds = 6; l_ts = 5; break;
    default:
      break;
    }

    int t = 0;

    for (int i = 0; i < l_dims; i++)
    { inst[t] = dims[i]; t++; }
    for (int i = 0; i < l_lds; i++)
    { inst[t] = lds[i]; t++; }
    for (int i = 0; i < l_ts; i++)
    { inst[t] = (int) ts[i]; t++; }

    return t;
  }

  __host__ unsigned long long int getFops () const
  {
    unsigned long long int accum = 0;
    switch (op_type)
    {
    case nop:
      break;
    case getrf:
      for (unsigned long long int x = dims[0], y = dims[1]; x > 0 && y > 0; x--, y--)
      { accum += (y - 1) + 2 * (x - 1) * (y - 1); }
      break;
    case trsml: case trsml_lr:
      for (unsigned long long int x = dims[2], y = dims[1], x_b = dims[0]; x > 0 && y > 0; x--, y--)
      { accum += 2 * (y - 1) * x_b; }
      break;
    case trsmr: case trsmr_lr:
      for (unsigned long long int x = dims[0], y = dims[2], y_b = dims[1];  x > 0 && y > 0; x--, y--)
      { accum += y_b + 2 * (x - 1) * y_b; }
      break;
    case gemm:
      accum = dims[0];
      accum *= dims[1];
      accum *= dims[2];
      accum *= 2;
      break;
    case pivot:
      accum = 0;
      break;
    case gemm3:
    {
      unsigned long long int c1 = dims[1]; c1 *= dims[2]; c1 *= dims[3]; c1 *= 2;
      unsigned long long int c2 = dims[1]; c2 *= dims[0]; c2 *= dims[2]; c2 *= 2;
      accum = c1 + c2;
      break;
    }
    case gemm4:
    {
      unsigned long long int c1 = dims[1]; c1 *= dims[3]; c1 *= dims[4]; c1 *= 2;
      unsigned long long int c2 = dims[1]; c2 *= dims[2]; c2 *= dims[3]; c2 *= 2;
      unsigned long long int c3 = dims[1]; c3 *= dims[0]; c3 *= dims[2]; c3 *= 2;
      accum = c1 + c2 + c3;
      break;
    }
    case gemm5:
    {
      unsigned long long int c1 = dims[1]; c1 *= dims[4]; c1 *= dims[5]; c1 *= 2;
      unsigned long long int c2 = dims[1]; c2 *= dims[3]; c2 *= dims[4]; c2 *= 2;
      unsigned long long int c3 = dims[1]; c3 *= dims[2]; c3 *= dims[3]; c3 *= 2;
      unsigned long long int c4 = dims[1]; c4 *= dims[0]; c4 *= dims[2]; c4 *= 2;
      accum = c1 + c2 + c3 + c4;
      break;
    }
    }
    return accum;
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
      printf("GEMM "); 
      wr[0].printShort(); printf(" (%d x %d by %d) ", dims[0], dims[1], lds[0]);
      r[0].printShort(); if (ts[0]) { printf("T"); } printf(" (%d x %d by %d) ", dims[0], dims[2], lds[1]);
      r[1].printShort(); if (ts[1]) { printf("T"); } printf(" (%d x %d by %d) ", dims[2], dims[1], lds[2]);
      break;
    case pivot:
      printf("PIVOT ");
      wr[0].printShort(); printf(" (%d x %d by %d) ", dims[1], dims[0], lds[0]);
      r[0].printShort(); if (ts[0]) { printf("RECOVERY"); } else { printf("APPLY"); }
      break;
    case trsml_lr:
      printf("TRSML-LR ");
      wr[0].printShort(); if (ts[0]) { printf("T"); } printf(" (%d x %d by %d) ", dims[1], dims[0], lds[0]);
      r[0].printShort(); printf(" (%d x %d by %d) ", dims[1], dims[2], lds[1]);
      break;
    case trsmr_lr:
      printf("TRSMR-LR ");
      wr[0].printShort(); if (ts[0]) { printf("T"); } printf(" (%d x %d by %d) ", dims[1], dims[0], lds[0]);
      r[0].printShort(); printf(" (%d x %d by %d) ", dims[2], dims[0], lds[1]);
      break;
    case gemm3:
      printf("GEMM3 ");
      wr[0].printShort(); printf(" (%d x %d by %d) ", dims[0], dims[1], lds[0]);
      r[0].printShort(); if (ts[0]) { printf("T"); } printf(" (%d x %d by %d) ", dims[0], dims[2], lds[1]);
      r[1].printShort(); if (ts[1]) { printf("T"); } printf(" (%d x %d by %d) ", dims[2], dims[3], lds[2]);
      r[2].printShort(); if (ts[2]) { printf("T"); } printf(" (%d x %d by %d) ", dims[3], dims[1], lds[3]);
      break;
    case gemm4:
      printf("GEMM4 ");
      wr[0].printShort(); printf(" (%d x %d by %d) ", dims[0], dims[1], lds[0]);
      r[0].printShort(); if (ts[0]) { printf("T"); } printf(" (%d x %d by %d) ", dims[0], dims[2], lds[1]);
      r[1].printShort(); if (ts[1]) { printf("T"); } printf(" (%d x %d by %d) ", dims[2], dims[3], lds[2]);
      r[2].printShort(); if (ts[2]) { printf("T"); } printf(" (%d x %d by %d) ", dims[3], dims[4], lds[3]);
      r[3].printShort(); if (ts[3]) { printf("T"); } printf(" (%d x %d by %d) ", dims[4], dims[1], lds[4]);
      break;
    case gemm5:
      printf("GEMM5 ");
      wr[0].printShort(); printf(" (%d x %d by %d) ", dims[0], dims[1], lds[0]);
      r[0].printShort(); if (ts[0]) { printf("T"); } printf(" (%d x %d by %d) ", dims[0], dims[2], lds[1]);
      r[1].printShort(); if (ts[1]) { printf("T"); } printf(" (%d x %d by %d) ", dims[2], dims[3], lds[2]);
      r[2].printShort(); if (ts[2]) { printf("T"); } printf(" (%d x %d by %d) ", dims[3], dims[4], lds[3]);
      r[3].printShort(); if (ts[3]) { printf("T"); } printf(" (%d x %d by %d) ", dims[4], dims[5], lds[4]);
      r[4].printShort(); if (ts[4]) { printf("T"); } printf(" (%d x %d by %d) ", dims[5], dims[1], lds[5]);
      break;
    }

    printf("{fp-ops: %llu}\n", getFops());
  }


  
};

class h_ops_tree : public h_ops
{
private:

  h_ops_tree * next;
  h_ops_tree * child;

public:

  __host__ h_ops_tree (const operation_t op_in = nop) : h_ops (op_in)
  {
    next = nullptr;
    child = nullptr;
  }

  __host__ h_ops_tree (const operation_t op_in, const h_index * M, const int nx, const int ny, const int ld) : 
    h_ops (op_in, M, nx, ny, ld)
  {
    next = nullptr;
    child = nullptr;
  }

  __host__ h_ops_tree (const operation_t op_in, const h_index * B, const h_index * M, const int nx_b, const int ny_b, const int dim_m, 
    const int ld_b, const int ld_m) :
    h_ops (op_in, B, M, nx_b, ny_b, dim_m, ld_b, ld_m)
  {
    next = nullptr;
    child = nullptr;
  }

  __host__ h_ops_tree (const operation_t op_in, const h_index * M, const h_index * A, const h_index * B, const int m, const int n, const int k, 
    const int ld_m, const int ld_a, const int ld_b, const bool A_T, const bool B_T) :
    h_ops (op_in, M, A, B, m, n, k, ld_m, ld_a, ld_b, A_T, B_T)
  {
    next = nullptr;
    child = nullptr;
  }

  __host__ h_ops_tree (const operation_t op_in, const h_index * M, const h_index * P, const int nx, const int ny, const int ld, const bool p_T) :
    h_ops (op_in, M, P, nx, ny, ld, p_T)
  {
    next = nullptr;
    child = nullptr;
  }

  __host__ h_ops_tree (const operation_t op_in, const h_index * LR, const h_index * M, const int nx_lr, const int ny_lr, const int dim_m,
    const int ld_lr, const int ld_m, const bool lr_T) :
    h_ops (op_in, LR, M, nx_lr, ny_lr, dim_m, ld_lr, ld_m, lr_T)
  {
    next = nullptr;
    child = nullptr;
  }

  __host__ h_ops_tree (const operation_t op_in, const h_index * M, const h_index * A, const h_index * B, const h_index * C,
    const int m, const int n, const int k, const int l, const int ld_m, const int ld_a, const int ld_b, const int ld_c,
    const bool a_T, const bool b_T, const bool c_T) :
    h_ops (op_in, M, A, B, C, m, n, k, l, ld_m, ld_a, ld_b, ld_c, a_T, b_T, c_T)
  {
    next = nullptr;
    child = nullptr;
  }

  __host__ h_ops_tree (const operation_t op_in, const h_index * M, const h_index * A, const h_index * B, const h_index * C, const h_index * D,
    const int m, const int n, const int k, const int l, const int o, const int ld_m, const int ld_a, const int ld_b, const int ld_c, const int ld_d,
    const bool a_T, const bool b_T, const bool c_T, const bool d_T) :
    h_ops (op_in, M, A, B, C, D, m, n, k, l, o, ld_m, ld_a, ld_b, ld_c, ld_d, a_T, b_T, c_T, d_T)
  {
    next = nullptr;
    child = nullptr;
  }

  __host__ h_ops_tree (const operation_t op_in, const h_index * M, const h_index * A, const h_index * B, const h_index * C, const h_index * D, const h_index * E, 
    const int m, const int n, const int k, const int l, const int o, const int p,
    const int ld_m, const int ld_a, const int ld_b, const int ld_c, const int ld_d, const int ld_e,
    const bool a_T, const bool b_T, const bool c_T, const bool d_T, const bool e_T) :
    h_ops (op_in, M, A, B, C, D, E, m, n, k, l, o, p, ld_m, ld_a, ld_b, ld_c, ld_d, ld_e, a_T, b_T, c_T, d_T, e_T)
  {
    next = nullptr;
    child = nullptr;
  }

  __host__ ~h_ops_tree ()
  {
    delete child;
    delete next;
  }

  __host__ h_ops_tree * getNext () const
  {
    return next;
  }

  __host__ void hookup_next (h_ops_tree *tree)
  {
    if (next != nullptr)
    { next -> hookup_next (tree); }
    else
    { next = tree; }
  }

  __host__ void hookup_child (h_ops_tree *tree)
  {
    if (child != nullptr)
    { child -> hookup_next (tree); }
    else
    { child = tree; }
  }

  __host__ int length_child() const
  {
    return (child == nullptr) ? 1 : child -> length();
  }

  __host__ int length() const
  {
    int length = length_child();
    for (h_ops_tree *ptr = next; ptr != nullptr; ptr = ptr -> next)
    { length += ptr -> length_child(); }
    return length;
  }

  __host__ h_ops_tree * clone() const
  {
    switch (op_type)
    {
    case nop:
      return new h_ops_tree ();
    case getrf:
      return new h_ops_tree (op_type, &wr[0], dims[0], dims[1], lds[0]);
    case trsml: case trsmr:
      return new h_ops_tree (op_type, &wr[0], &r[0], dims[0], dims[1], dims[2], lds[0], lds[1]);
    case gemm:
      return new h_ops_tree (op_type, &wr[0], &r[0], &r[1], dims[0], dims[1], dims[2], lds[0], lds[1], lds[2], (bool)ts[0], (bool)ts[1]);
    case pivot:
      return new h_ops_tree (op_type, &wr[0], &r[0], dims[0], dims[1], lds[0], (bool)ts[0]);
    case trsml_lr: case trsmr_lr:
      return new h_ops_tree (op_type, &wr[0], &r[0], dims[0], dims[1], dims[2], lds[0], lds[1], (bool)ts[0]);
    case gemm3:
      return new h_ops_tree (op_type, &wr[0], &r[0], &r[1], &r[2], dims[0], dims[1], dims[2], dims[3],
        lds[0], lds[1], lds[2], lds[3], (bool)ts[0], (bool)ts[1], (bool)ts[2]);
    case gemm4:
      return new h_ops_tree (op_type, &wr[0], &r[0], &r[1], &r[2], &r[3], dims[0], dims[1], dims[2], dims[3], dims[4], 
        lds[0], lds[1], lds[2], lds[3], lds[4], (bool)ts[0], (bool)ts[1], (bool)ts[2], (bool)ts[3]);
    case gemm5:
      return new h_ops_tree (op_type, &wr[0], &r[0], &r[1], &r[2], &r[3], &r[4], dims[0], dims[1], dims[2], dims[3], dims[4], dims[5],
        lds[0], lds[1], lds[2], lds[3], lds[4], lds[5], (bool)ts[0], (bool)ts[1], (bool)ts[2], (bool)ts[3], (bool)ts[4]);
    default:
      return nullptr;
    }
  }

  __host__ h_ops_tree * flatten () const
  {
    h_ops_tree * list = (child == nullptr) ? clone() : child -> flatten();
    if (next != nullptr)
    { list -> hookup_next (next -> flatten()); }
    return list;
  }

  __host__ unsigned long long int getFops() const
  {
    if (child == nullptr)
    { return h_ops::getFops(); }
    else
    { return child -> getFops_All(); }
  }

  __host__ unsigned long long int getFops_All() const
  {
    unsigned long long int accum = 0;
    for (const h_ops_tree * op_ptr = this; op_ptr != nullptr; op_ptr = op_ptr -> next)
    { accum += op_ptr -> getFops(); }
    return accum;
  }

  __host__ void print (const int op_id = 0, const int indent = 0) const
  {
    for (int i = 0; i < indent; i++) { printf("  "); }

    if (child == nullptr) { printf("%d: ", op_id); }

    h_ops::print();

    if (child != nullptr) { child -> print(op_id, indent + 1); }

    if (next != nullptr) 
    {
      const int l_this = (child == nullptr) ? 1 : child -> length();
      next -> print(op_id + l_this, indent); 
    }
  }

};

#endif