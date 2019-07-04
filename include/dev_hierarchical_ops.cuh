
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

  long long int flops;

public:

  __host__ h_ops ()
  {
    op_type = nop;
    n_rw = 0;
    read_and_write = nullptr;
    n_ro = 0;
    read_only = nullptr;
    flops = 0;
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
    flops = 0;
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
    flops = 0;
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
    flops = 0;
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
        case diff_mat: case same_mat_diff_branch: case same_node_no_overlap: case same_node_different_temp:
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
        case diff_mat: case same_mat_diff_branch: case same_node_no_overlap: case same_node_different_temp:
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
      case diff_mat: case same_mat_diff_branch: case same_node_no_overlap: case same_node_different_temp:
        break;
      case same_branch_diff_node: case same_node_overlapped: case same_index:
        dep |= (int) anti_dep;
      }
    }

    return (dependency_t) dep;
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
      { 
        printf("Error: GETRF on incompatible block.\n");
        inst[0] = (int) nop;
        return nop_l - 2;  
      }

      inst[0] = (int) getrf;
      inst[1] = M;
      inst[2] = offset_m;
      inst[3] = nx; 
      inst[4] = ny; 
      inst[5] = ld;
      return getrf_l - 2;
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
        ny_b = read_and_write[0].getNy(read_only[0].getNy());
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
        ny_b = read_and_write[0].getNy(read_only[0].getNy());
        nx_l = read_only[0].getNx();
        ld_b = read_and_write[0].getLd_x();
        ld_l = read_only[0].getLd_x();
        b_T = read_and_write[0].getTranspose();
      }
      else
      { 
        printf("Error: TRSML on incompatible block.\n");
        inst[0] = (int) nop;
        return nop_l - 2;
      }

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
      return trsml_l - 2;
    }
    case trsmr:
    {
      int B, U, offset_b, offset_u, nx_b, ny_b, ny_u, ld_b, ld_u, b_T;

      if (read_and_write[0].isVT() && read_only[0].isDense())
      {
        B = mapping[1];
        U = mapping[2];
        offset_b = read_and_write[0].getOffset_x();
        offset_u = read_only[0].getOffset_x();
        nx_b = read_and_write[0].getNx(read_only[0].getNx());
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
        nx_b = read_and_write[0].getNx(read_only[0].getNx());
        ny_b = read_and_write[0].getNy();
        ny_u = read_only[0].getNy();
        ld_b = read_and_write[0].getLd_x();
        ld_u = read_only[0].getLd_x();
        b_T = read_and_write[0].getTranspose();
      }
      else
      { 
        printf("Error: TRSMR on incompatible block.\n");
        inst[0] = (int) nop;
        return nop_l - 2;
      }

      inst[0] = (int) trsmr;
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
      return trsmr_l - 2;
    }
    case gemm:
    {
      int M, A, B, offset_m, offset_a, offset_b, m, n, k, ld_m, ld_a, ld_b, a_T, b_T;
      bool gemm_write = false;

      if (read_and_write[0].isU() && read_only[0].isDense() && read_only[1].isU())
      {
        gemm_write = true;

        M = mapping[0];
        A = mapping[2];
        B = mapping[3];
        offset_m = read_and_write[0].getOffset_y();
        offset_a = read_only[0].getOffset();
        offset_b = read_only[1].getOffset_y();
        m = read_and_write[0].getNy(read_only[0].getNy());
        n = read_and_write[0].getRank(read_only[1].getRank());
        k = read_only[0].getNx(read_only[1].getNy());
        ld_m = read_and_write[0].getLd_y();
        ld_a = read_only[0].getLd_x();
        ld_b = read_only[1].getLd_y();
        a_T = read_only[0].getTranspose();
        b_T = read_only[1].getTranspose();
      }
      else if (read_and_write[0].isVT() && read_only[0].isVT() && read_only[1].isDense())
      {
        gemm_write = true;

        M = mapping[1];
        A = mapping[4];
        B = mapping[3];
        offset_m = read_and_write[0].getOffset_x();
        offset_a = read_only[1].getOffset();
        offset_b = read_only[0].getOffset_x();
        m = read_and_write[0].getNx(read_only[1].getNx());
        n = read_and_write[0].getRank(read_only[0].getRank());
        k = read_only[1].getNy(read_only[0].getNx());
        ld_m = read_and_write[0].getLd_x();
        ld_a = read_only[1].getLd_x();
        ld_b = read_only[0].getLd_x();
        a_T = !read_only[1].getTranspose();
        b_T = !read_only[0].getTranspose();
      }
      else if (read_and_write[0].isDense() && read_only[0].isDense() && read_only[1].isDense())
      {
        gemm_write = true;

        M = mapping[0];
        A = mapping[1];
        B = mapping[2];
        offset_m = read_and_write[0].getOffset();
        offset_a = read_only[0].getOffset();
        offset_b = read_only[1].getOffset();
        m = read_and_write[0].getNy(read_only[0].getNy());
        n = read_and_write[0].getNx(read_only[1].getNx());
        k = read_only[0].getNy(read_only[1].getNy());
        ld_m = read_and_write[0].getLd_x();
        ld_a = read_only[0].getLd_x();
        ld_b = read_only[1].getLd_x();
        a_T = read_only[0].getTranspose();
        b_T = read_only[1].getTranspose();
      }

      int C, offset_c, l, ld_c, c_T;

      if (gemm_write)
      {
        inst[0] = (int) gemm;
        inst[1] = M;
        inst[2] = A;
        inst[3] = B;
        inst[4] = offset_m;
        inst[5] = offset_a;
        inst[6] = offset_b;
        inst[7] = m;
        inst[8] = n;
        inst[9] = k;
        inst[10] = ld_m;
        inst[11] = ld_a;
        inst[12] = ld_b;
        inst[13] = a_T;
        inst[14] = b_T;
        return gemm_l - 2;
      }
      else if (read_and_write[0].isU() && read_only[0].isLowRank() && read_only[1].isU())
      {
        gemm_write = true;

        M = mapping[0];
        A = mapping[2];
        B = mapping[3];
        C = mapping[4];
        offset_m = read_and_write[0].getOffset_y();
        offset_a = read_only[0].getOffset_y();
        offset_b = read_only[0].getOffset_x();
        offset_c = read_only[1].getOffset_y();
        m = read_and_write[0].getNy(read_only[0].getNy());
        n = read_and_write[0].getRank(read_only[1].getRank());
        k = read_only[0].getRank();
        l = read_only[0].getNx(read_only[1].getNy());
        ld_m = read_and_write[0].getLd_y();
        ld_a = read_only[0].getLd_y();
        ld_b = read_only[0].getLd_x();
        ld_c = read_only[1].getLd_y();
        a_T = read_only[0].getTranspose();
        b_T = !read_only[0].getTranspose();
        c_T = read_only[1].getTranspose();
      }
      else if (read_and_write[0].isVT() && read_only[0].isVT() && read_only[1].isLowRank())
      {
        gemm_write = true;

        M = mapping[1];
        A = mapping[5];
        B = mapping[4];
        C = mapping[3];
        offset_m = read_and_write[0].getOffset_x();
        offset_a = read_only[1].getOffset_x();
        offset_b = read_only[1].getOffset_y();
        offset_c = read_only[0].getOffset_x();
        m = read_and_write[0].getNx(read_only[1].getNx());
        n = read_and_write[0].getRank(read_only[0].getRank());
        k = read_only[1].getRank();
        l = read_only[0].getNy(read_only[1].getNx());
        ld_m = read_and_write[0].getLd_x();
        ld_a = read_only[1].getLd_x();
        ld_b = read_only[1].getLd_y();
        ld_c = read_only[0].getLd_x();
        a_T = read_only[1].getTranspose();
        b_T = !read_only[1].getTranspose();
        c_T = !read_only[0].getTranspose();
      }
      else if (read_and_write[0].isDense() && read_only[0].isLowRank() && read_only[1].isDense())
      {
        gemm_write = true;

        M = mapping[0];
        A = mapping[1];
        B = mapping[2];
        C = mapping[3];
        offset_m = read_and_write[0].getOffset();
        offset_a = read_only[0].getOffset_y();
        offset_b = read_only[0].getOffset_x();
        offset_c = read_only[1].getOffset();
        m = read_and_write[0].getNy(read_only[0].getNy());
        n = read_and_write[0].getNx(read_only[1].getNx());
        k = read_only[0].getRank();
        l = read_only[0].getNy(read_only[1].getNy());
        ld_m = read_and_write[0].getLd_x();
        ld_a = read_only[0].getLd_y();
        ld_b = read_only[0].getLd_x();
        ld_c = read_only[1].getLd_x();
        a_T = read_only[0].getTranspose();
        b_T = !read_only[0].getTranspose();
        c_T = read_only[1].getTranspose();
      }
      else if (read_and_write[0].isDense() && read_only[0].isDense() && read_only[1].isLowRank())
      {
        gemm_write = true;

        M = mapping[0];
        A = mapping[1];
        B = mapping[2];
        C = mapping[3];
        offset_m = read_and_write[0].getOffset();
        offset_a = read_only[0].getOffset();
        offset_b = read_only[1].getOffset_y();
        offset_c = read_only[1].getOffset_x();
        m = read_and_write[0].getNy(read_only[0].getNy());
        n = read_and_write[0].getNx(read_only[1].getNx());
        k = read_only[0].getNy(read_only[1].getNy());
        l = read_only[1].getRank();
        ld_m = read_and_write[0].getLd_x();
        ld_a = read_only[0].getLd_x();
        ld_b = read_only[1].getLd_y();
        ld_c = read_only[1].getLd_x();
        a_T = read_only[0].getTranspose();
        b_T = read_only[1].getTranspose();
        c_T = !read_only[1].getTranspose();
      }

      int D, offset_d, o, ld_d, d_T;

      if (gemm_write)
      {
        int t_size1;
        int control = getControl_GEMM_3x(&t_size1, m, n, k, l);

        inst[0] = (int) gemm_3x;
        inst[1] = M;
        inst[2] = A;
        inst[3] = B;
        inst[4] = C;
        inst[5] = offset_m;
        inst[6] = offset_a;
        inst[7] = offset_b;
        inst[8] = offset_c;
        inst[9] = m;
        inst[10] = n;
        inst[11] = k;
        inst[12] = l;
        inst[13] = ld_m;
        inst[14] = ld_a;
        inst[15] = ld_b;
        inst[16] = ld_c;
        inst[17] = a_T;
        inst[18] = b_T;
        inst[19] = c_T;
        inst[20] = control;
        inst[21] = t_size1;

        return gemm_3x_l - 2;
      }
      else if (read_and_write[0].isDense() && read_only[0].isLowRank() && read_only[1].isLowRank())
      {
        gemm_write = true;

        M = mapping[0];
        A = mapping[1];
        B = mapping[2];
        C = mapping[3];
        D = mapping[4];
        offset_m = read_and_write[0].getOffset();
        offset_a = read_only[0].getOffset_y();
        offset_b = read_only[0].getOffset_x();
        offset_c = read_only[1].getOffset_y();
        offset_d = read_only[1].getOffset_x();
        m = read_and_write[0].getNy(read_only[0].getNy());
        n = read_and_write[0].getNx(read_only[1].getNx());
        k = read_only[0].getRank();
        l = read_only[0].getNx(read_only[1].getNy());
        o = read_only[1].getRank();
        ld_m = read_and_write[0].getLd_x();
        ld_a = read_only[0].getLd_y();
        ld_b = read_only[0].getLd_x();
        ld_c = read_only[1].getLd_y();
        ld_d = read_only[1].getLd_x();
        a_T = read_only[0].getTranspose();
        b_T = !read_only[0].getTranspose();
        c_T = read_only[1].getTranspose();
        d_T = !read_only[1].getTranspose();
      }


      if (gemm_write)
      {
        int t_size1, t_size2;
        int control = getControl_GEMM_4x (&t_size1, &t_size2, m, n, k, l, o);

        inst[0] = (int) gemm_4x;
        inst[1] = M;
        inst[2] = A;
        inst[3] = B;
        inst[4] = C;
        inst[5] = D;
        inst[6] = offset_m;
        inst[7] = offset_a;
        inst[8] = offset_b;
        inst[9] = offset_c;
        inst[10] = offset_d;
        inst[11] = m;
        inst[12] = n;
        inst[13] = k;
        inst[14] = l;
        inst[15] = o;
        inst[16] = ld_m;
        inst[17] = ld_a;
        inst[18] = ld_b;
        inst[19] = ld_c;
        inst[20] = ld_d;
        inst[21] = a_T;
        inst[22] = b_T;
        inst[23] = c_T;
        inst[24] = d_T;
        inst[25] = control;
        inst[26] = t_size1;
        inst[27] = t_size2;

        return gemm_4x_l - 2;
      }
      else
      {
        printf("Error: GEMM on incompatible block.\n"); print();
        inst[0] = (int) nop;
        return nop_l - 2;
      }

    }
    case accum:
    {
      if (read_and_write[0].isDense() && read_only[0].isLowRank())
      {
        int M, A, B, offset_m, offset_a, offset_b, m, n, k, ld_m, ld_a, ld_b, a_T, b_T;
        M = mapping[0];
        A = mapping[1];
        B = mapping[2];
        offset_m = read_and_write[0].getOffset();
        offset_a = read_only[0].getOffset_y();
        offset_b = read_only[0].getOffset_x();
        m = read_and_write[0].getNy(read_only[0].getNy());
        n = read_and_write[0].getNx(read_only[0].getNx());
        k = read_only[0].getRank();
        ld_m = read_and_write[0].getLd_x();
        ld_a = read_only[0].getLd_y();
        ld_b = read_only[0].getLd_x();
        a_T = read_only[0].getTranspose();
        b_T = !read_only[0].getTranspose();

        inst[0] = (int) gemm_plus;
        inst[1] = M;
        inst[2] = A;
        inst[3] = B;
        inst[4] = offset_m;
        inst[5] = offset_a;
        inst[6] = offset_b;
        inst[7] = m;
        inst[8] = n;
        inst[9] = k;
        inst[10] = ld_m;
        inst[11] = ld_a;
        inst[12] = ld_b;
        inst[13] = a_T;
        inst[14] = b_T;
        return gemm_plus_l - 2;
      }
      else if (read_and_write[0].isLowRank() && read_only[0].isLowRank())
      {
        int U1, VT1, U2, VT2, offset_u1, offset_vt1, offset_u2, offset_vt2, nx, ny, rank1, rank2, ld_u1, ld_vt1, ld_u2, ld_vt2;

        U1 = mapping[0];
        VT1 = mapping[1];
        U2 = mapping[2];
        VT2 = mapping[3];
        offset_u1 = read_and_write[0].getOffset_y();
        offset_vt1 = read_and_write[0].getOffset_x();
        offset_u2 = read_only[0].getOffset_y();
        offset_vt2 = read_only[0].getOffset_x();
        nx = read_and_write[0].getNx(read_only[0].getNx());
        ny = read_and_write[0].getNy(read_only[0].getNy());
        rank1 = read_and_write[0].getRank();
        rank2 = read_only[0].getRank();
        ld_u1 = read_and_write[0].getLd_y();
        ld_vt1 = read_and_write[0].getLd_x();
        ld_u2 = read_only[0].getLd_y();
        ld_vt2 = read_only[0].getLd_x();

        inst[0] = (int) accum;
        inst[1] = U1;
        inst[2] = VT1;
        inst[3] = U2;
        inst[4] = VT2;
        inst[5] = offset_u1;
        inst[6] = offset_vt1;
        inst[7] = offset_u2;
        inst[8] = offset_vt2;
        inst[9] = nx;
        inst[10] = ny;
        inst[11] = rank1;
        inst[12] = rank2;
        inst[13] = ld_u1;
        inst[14] = ld_vt1;
        inst[15] = ld_u2;
        inst[16] = ld_vt2;

        return accum_l - 2;
      }
      else if (read_and_write[0].isLowRank() && read_only[0].isLowRank())
      {
        // TODO
        printf("Error: Accum dense awaiting implementation.\n");
        return 0;
      }
      else
      {
        printf("Error: ACCUM on incompatible block.\n");
        inst[0] = (int) nop;
        return nop_l - 2;
      }
    }
    default:
    { 
      inst[0] = (int) nop;
      return nop_l - 2;
    }
    }


  }

  __host__ long long int getFlops ()
  {
    if (flops > 0)
    { return flops; }
    else
    {
      switch (opType())
      {
      case getrf:
      {
        long long int nx = 0, ny = 0;

        if (read_and_write[0].isDense())
        {
          nx = read_and_write[0].getNx();
          ny = read_and_write[0].getNy();
          flops = getFlops_GETRF(nx, ny);
        }
        else
        { flops = 0; }
        break;
      }
      case trsml:
      {
        long long int nx_b = 0, ny_b = 0, nx_l = 0;

        if (read_and_write[0].isU() && read_only[0].isDense())
        {
          nx_b = read_and_write[0].getRank();
          ny_b = read_and_write[0].getNy(read_only[0].getNy());
          nx_l = read_only[0].getNx();
          flops = getFlops_TRSML(nx_b, ny_b, nx_l);
        }
        else if (read_and_write[0].isDense() && read_only[0].isDense())
        {
          nx_b = read_and_write[0].getNx();
          ny_b = read_and_write[0].getNy(read_only[0].getNy());
          nx_l = read_only[0].getNx();
          flops = getFlops_TRSML(nx_b, ny_b, nx_l);
        }
        else
        { flops = 0; }
        break;
      }
      case trsmr:
      {
        long long int nx_b = 0, ny_b = 0, ny_u = 0;

        if (read_and_write[0].isVT() && read_only[0].isDense())
        {
          nx_b = read_and_write[0].getNx(read_only[0].getNx());
          ny_b = read_and_write[0].getRank();
          ny_u = read_only[0].getNy();
          flops = getFlops_TRSMR(nx_b, ny_b, ny_u);
        }
        else if (read_and_write[0].isDense() && read_only[0].isDense())
        {
          nx_b = read_and_write[0].getNx(read_only[0].getNx());
          ny_b = read_and_write[0].getNy();
          ny_u = read_only[0].getNy();
          flops = getFlops_TRSMR(nx_b, ny_b, ny_u);
        }
        else
        { flops = 0; }
        break;
      }
      case gemm:
      {
        long long int m = 0, n = 0, k = 0;
        bool gemm_write = false;

        if (read_and_write[0].isU() && read_only[0].isDense() && read_only[1].isU())
        {
          gemm_write = true;

          m = read_and_write[0].getNy(read_only[0].getNy());
          n = read_and_write[0].getRank(read_only[1].getRank());
          k = read_only[0].getNx(read_only[1].getNy());
        }
        else if (read_and_write[0].isVT() && read_only[0].isVT() && read_only[1].isDense())
        {
          gemm_write = true;

          m = read_and_write[0].getNx(read_only[1].getNx());
          n = read_and_write[0].getRank(read_only[0].getRank());
          k = read_only[1].getNy(read_only[0].getNx());
        }
        else if (read_and_write[0].isDense() && read_only[0].isDense() && read_only[1].isDense())
        {
          gemm_write = true;

          m = read_and_write[0].getNy(read_only[0].getNy());
          n = read_and_write[0].getNx(read_only[1].getNx());
          k = read_only[0].getNy(read_only[1].getNy());
        }

        long long int l = 0;

        if (gemm_write)
        { flops = getFlops_GEMM(m, n, k); break; }
        else if (read_and_write[0].isU() && read_only[0].isLowRank() && read_only[1].isU())
        {
          gemm_write = true;

          m = read_and_write[0].getNy(read_only[0].getNy());
          n = read_and_write[0].getRank(read_only[1].getRank());
          k = read_only[0].getRank();
          l = read_only[0].getNx(read_only[1].getNy());
        }
        else if (read_and_write[0].isVT() && read_only[0].isVT() && read_only[1].isLowRank())
        {
          gemm_write = true;

          m = read_and_write[0].getNx(read_only[1].getNx());
          n = read_and_write[0].getRank(read_only[0].getRank());
          k = read_only[1].getRank();
          l = read_only[0].getNy(read_only[1].getNx());
        }
        else if (read_and_write[0].isDense() && read_only[0].isLowRank() && read_only[1].isDense())
        {
          gemm_write = true;

          m = read_and_write[0].getNy(read_only[0].getNy());
          n = read_and_write[0].getNx(read_only[1].getNx());
          k = read_only[0].getRank();
          l = read_only[0].getNy(read_only[1].getNy());
        }
        else if (read_and_write[0].isDense() && read_only[0].isDense() && read_only[1].isLowRank())
        {
          gemm_write = true;

          m = read_and_write[0].getNy(read_only[0].getNy());
          n = read_and_write[0].getNx(read_only[1].getNx());
          k = read_only[0].getNy(read_only[1].getNy());
          l = read_only[1].getRank();
        }

        long long int o = 0;

        if (gemm_write)
        { flops = getFlops_GEMM_3x(m, n, k, l); break; }
        else if (read_and_write[0].isDense() && read_only[0].isLowRank() && read_only[1].isLowRank())
        {
          gemm_write = true;

          m = read_and_write[0].getNy(read_only[0].getNy());
          n = read_and_write[0].getNx(read_only[1].getNx());
          k = read_only[0].getRank();
          l = read_only[0].getNx(read_only[1].getNy());
          o = read_only[1].getRank();
        }

        if (gemm_write)
        { flops = getFlops_GEMM_4x(m, n, k, l, o); }
        else
        { flops = 0; }
        break;

      }
      case accum:
      {
        if (read_and_write[0].isDense() && read_only[0].isLowRank())
        {
          long long int m = 0, n = 0, k = 0;

          m = read_and_write[0].getNy(read_only[0].getNy());
          n = read_and_write[0].getNx(read_only[0].getNx());
          k = read_only[0].getRank();

          flops = getFlops_GEMM(m, n, k);
        }
        else if (read_and_write[0].isLowRank() && read_only[0].isLowRank())
        {
          long long int nx = 0, ny = 0, rank1 = 0, rank2 = 0;

          nx = read_and_write[0].getNx(read_only[0].getNx());
          ny = read_and_write[0].getNy(read_only[0].getNy());
          rank1 = read_and_write[0].getRank();
          rank2 = read_only[0].getRank();

          flops = getFlops_LrAccum(nx, ny, rank1, rank2);
        }
        else if (read_and_write[0].isLowRank() && read_only[0].isLowRank())
        {
          // TODO
          flops = 0;
        }
        else
        { flops = 0; }
        break;
      }
      default:
      { flops = 0; }
      }

      return flops;
    }
  }

  __host__ static int getControl_GEMM_3x (int * t_size, const int m, const int n, const int k, const int l)
  {
    const int size_1 = m * l, size_2 = n * k;

    bool b_ab_a = size_1 * (k + n) <= size_2 * (m + l);
    * t_size = b_ab_a ? size_1 : size_2;

    return (int) b_ab_a; 
  }

  __host__ static int getControl_GEMM_4x (int * t_size1, int * t_size2, const int m, const int n, const int k, const int l, const int o)
  {
    const int size_1 = m * l, size_2 = m * o, size_3 = n * k, size_4 = n * l, size_5 = k * o;

    const int f_ab = size_1 * k, f_bc = size_5 * l, f_cd = size_4 * o;
    const bool b_ab_bc = f_ab <= f_bc, b_bc_cd = f_bc <= f_cd;

    const int f_abc_d = b_ab_bc ? size_2 * (l + n) + f_ab : size_2 * (k + n) + f_bc;
    const int f_a_bcd = b_bc_cd ? size_3 * (o + m) + f_bc : size_3 * (l + m) + f_cd;
    const int f_ab_cd = f_ab + f_cd + size_1 * n;

    const bool b_abc_a = f_abc_d <= f_a_bcd, b_abc_ab = f_abc_d <= f_ab_cd;
    
    int control;
    if (b_abc_a && b_abc_ab) // (A x B x C) x D
    {
      * t_size2 = size_2;
      if (b_ab_bc) // ((A x B) x C) x D
      { control = 0; * t_size1 = size_1; }
      else // (A x (B x C)) x D
      { control = 1; * t_size1 = size_5; }
    }
    else if (b_abc_ab) // A x (B x C x D)
    {
      * t_size2 = size_3;
      if (b_bc_cd) // A x ((B x C) x D)
      { control = 2; * t_size1 = size_5; }
      else // A x (B x (C x D))
      { control = 3; * t_size1 = size_4; }
    }
    else // (A x B) x (C x D)
    { control = 4; * t_size1 = size_1; * t_size2 = size_4; }

    return control; 
  }

  __host__ static long long int getFlops_GETRF (const long long int nx, const long long int ny)
  {
    long long int accum = 0; const long long int n = nx > ny ? ny : nx;
    for (long long int i = 0; i < n; i++)
    { accum += (ny - i - 1) * (2 * (nx - i - 1) + 1); }
    return accum;
  }

  __host__ static long long int getFlops_TRSML (const long long int nx_b, const long long int ny_b, const long long int nx_l)
  {
    long long int accum = 0; const long long int n = nx_l > ny_b ? ny_b : nx_l;
    accum = (2 * ny_b - n - 1) * n * nx_b;
    return accum;
  }

  __host__ static long long int getFlops_TRSMR (const long long int nx_b, const long long int ny_b, const long long int ny_u)
  {
    long long int accum = 0; const long long int n = nx_b > ny_u ? ny_u : nx_b;
    accum = (2 * nx_b - n) * n * ny_b;
    return accum;
  }

  __host__ static long long int getFlops_GEMM (const long long int m, const long long int n, const long long int k)
  {
    long long int accum = m * n * k * 2;
    return accum;
  }

  __host__ static long long int getFlops_GEMM_3x (const long long int m, const long long int n, const long long int k, const long long int l)
  {
    long long int f1 = k * n * (m + l);
    long long int f2 = m * l * (k + n);
    return (f1 <= f2 ? f1 : f2) * 2;
  }

  __host__ static long long int getFlops_GEMM_4x (const long long int m, const long long int n, const long long int k, const long long int l, const long long int o)
  {
    if ((m <= k && m <= l) || (o <= k && o <= l))
    { return getFlops_GEMM_3x (m, o, k, l) + getFlops_GEMM (m, n, o); }
    else if ((n <= l && n <= o) || (k <= o && k <= l))
    { return getFlops_GEMM_3x (k, n, l, o) + getFlops_GEMM (m, n, k); }
    else
    { return getFlops_GEMM (m, l, k) + getFlops_GEMM (k, o, n) + getFlops_GEMM (m, l, n); }
  }

  __host__ static long long int getFlops_QR (const long long int nx, const long long int ny)
  {
    long long int accum = nx * nx * (3 * ny - nx) * 2;
    return accum;
  }

  __host__ static long long int getFlops_LrAccum (const long long int nx, const long long int ny, const long long int rank1, const long long int rank2)
  {
    long long int accum = getFlops_GEMM_3x(ny, rank1, rank1, nx) + getFlops_GEMM_3x(ny, rank2, rank2, nx);
    accum += getFlops_QR(rank1, ny);
    accum += getFlops_GEMM_3x(nx, rank1, rank1, ny) + getFlops_GEMM_3x(nx, rank2, rank2, ny);
    return accum;
  }

  __host__ void print() const
  {
    switch (opType())
    {
    case gemm:
    { printf("GEMM "); read_and_write[0].print(); read_only[0].print(); read_only[1].print(); printf("\n"); break; }
    case pivot:
    { printf("PVT "); read_and_write[0].print(); read_only[0].print(); printf("\n"); break; }
    case accum:
    { printf("ACCM "); read_and_write[0].print(); read_only[0].print(); printf("\n"); break; }
    case trsmr:
    { printf("TRSMR "); read_and_write[0].print(); read_only[0].print(); printf("\n"); break; }
    case trsml:
    { printf("TRSML "); read_and_write[0].print(); read_only[0].print(); printf("\n"); break; }
    case getrf:
    { printf("GETRF "); read_and_write[0].print(); printf("\n"); break; }
    default:
    { printf("NOP\n"); }
    }
    
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
      h_ops_tree * neo = new h_ops_tree [length_in];

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
      int length_ = 0;
#pragma omp parallel for reduction (+:length_) if (omp_in_parallel() == 0)
      for (int i = 0; i < l_children; i++) 
      { length_ += children[i].length(); }
      return length_;
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
      
      addr -> flops = flops;
      return addr;
    }
  }

  __host__ h_ops_tree * flatten (const int start_index = 0, const int length_max = 0, const int list_index = 0, h_ops_tree * list = nullptr) const
  {

    int length_ = 0, * lengths = new int [l_children];

#pragma omp parallel for reduction (+:length_) if (omp_in_parallel() == 0)
    for (int i = 0; i < l_children; i++) 
    { const int l = children[i].length(); lengths[i] = l; length_ += l; }

    if (length_ <= start_index)
    { delete[] lengths; return nullptr; }
    else
    { length_ = (length_max > 0 && length_max <= length_) ? length_max : length_; }

    if (list == nullptr)
    {
      list = clone (nullptr, false);
      list -> resizeChildren (length_);
    }

    int child_start = 0, child_end = l_children, insts_read = 0, insts_start = 0, end_length = 0;
    for (int i = 0; i < l_children; i++)
    {
      if (insts_read <= start_index)
      { child_start = i; insts_start = start_index - insts_read; }

      end_length = start_index + length_ - insts_read;

      if (end_length <= lengths[i])
      { child_end = i + 1; end_length = end_length > length_max ? length_max : end_length; break; }
      else
      { insts_read += lengths[i]; }
    }

    int iters = child_end - child_start;
    if (iters > 1)
    {
      int * work_index = new int [iters];
      work_index[0] = list_index;
      work_index[1] = list_index + lengths[child_start] - insts_start;

      for (int i = 1; i < iters - 1; i++)
      { work_index[i + 1] = work_index[i] + lengths[i + child_start]; }

#pragma omp parallel for if (omp_in_parallel() == 0)
      for (int i = child_start; i < child_end; i++)
      {
        const int index = work_index[i - child_start];
        if (children[i].l_children == 0)
        { children[i].clone(&(list -> children)[index], false); }
        else if (i == child_start)
        { children[i].flatten (insts_start, 0, index, list); }
        else if (i == child_end - 1)
        { children[i].flatten (0, end_length, index, list); }
        else
        { children[i].flatten (0, 0, index, list); }
      }
      delete[] work_index;
    }
    else
    {
      if (children[child_start].l_children == 0)
      { children[child_start].clone(&(list -> children)[list_index], false); }
      else
      { children[child_start].flatten (insts_start, end_length, list_index, list); }
    }

    delete[] lengths;
    return list;
  }

  __host__ long long int getFlops()
  {
    if (this == nullptr)
    { return 0; }
    else if (l_children == 0)
    { return h_ops::getFlops(); }
    else
    { 
      long long int accum = 0;
#pragma omp parallel for reduction (+:accum) if (omp_in_parallel == 0) 
      for (int i = 0; i < l_children; i++) 
      { accum += children[i].getFlops(); }
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