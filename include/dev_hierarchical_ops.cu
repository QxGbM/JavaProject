

#include <pspl.cuh>

h_ops::h_ops()
{
  op_type = nop;
  n_rw = 0;
  read_and_write = nullptr;
  n_ro = 0;
  read_only = nullptr;
  flops = 0;
}

h_ops::h_ops (const operation_t op_in, const h_index * M)
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

h_ops::h_ops (const operation_t op_in, const h_index * M1, const h_index * M2)
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

h_ops::h_ops (const operation_t op_in, const h_index * M1, const h_index * M2, const h_index * M3)
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

h_ops::~h_ops ()
{
  delete[] read_and_write;
  delete[] read_only;
}

operation_t h_ops::opType() const
{ return op_type; }

dependency_t h_ops::checkDependencyFrom (const h_ops * op_from) const
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

dependency_t h_ops::checkDependencyTo (const h_ops * op_to) const
{ return op_to -> checkDependencyFrom(this); }

int h_ops::getDataPointers (void ** data_ptrs, void ** tmp_ptrs) const
{
  int accum = 0;
  for (int i = 0; i < n_rw; i++)
  { accum += read_and_write[i].getDataPointers (&data_ptrs[accum], tmp_ptrs); }

  for (int i = 0; i < n_ro; i++)
  { accum += read_only[i].getDataPointers (&data_ptrs[accum], tmp_ptrs); }
    
  return accum;
}

int h_ops::writeOpParametersTo (int * inst, int * tmp_size, int * rnd_size, const int * mapping) const
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
      * tmp_size = 0;
      * rnd_size = 0;
      return nop_l;
    }

    inst[0] = (int) getrf;
    inst[1] = M;
    inst[2] = offset_m;
    inst[3] = nx; 
    inst[4] = ny; 
    inst[5] = ld;
    * tmp_size = 0;
    * rnd_size = 0;
    return getrf_l;
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
      * tmp_size = 0;
      * rnd_size = 0;
      return nop_l;
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
    * tmp_size = 0;
    * rnd_size = 0;
    return trsml_l;
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
      * tmp_size = 0;
      * rnd_size = 0;
      return nop_l;
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
    * tmp_size = 0;
    * rnd_size = 0;
    return trsmr_l;
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
      * tmp_size = 0;
      * rnd_size = 0;
      return gemm_l;
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
      int t_size;
      int control = getControl_GEMM_3x(&t_size, m, n, k, l);

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
      * tmp_size = t_size;
      * rnd_size = 0;
      return gemm_3x_l;
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
      int t_size, offset;
      int control = getControl_GEMM_4x (&t_size, &offset, m, n, k, l, o);

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
      inst[26] = offset;

      * tmp_size = t_size;
      * rnd_size = 0;
      return gemm_4x_l;
    }
    else
    {
      printf("Error: GEMM on incompatible block.\n"); print();
      inst[0] = (int) nop;
      * tmp_size = 0;
      * rnd_size = 0;
      return nop_l;
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
      * tmp_size = 0;
      * rnd_size = 0;
      return gemm_plus_l;
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

      int tmp, offset1, offset2;
      tmp = getTmpSize_ACCM_LR(&offset1, &offset2, nx, ny, rank1);

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
      inst[17] = offset1;
      inst[18] = offset2;

      * tmp_size = tmp;
      * rnd_size = nx * rank1;
      return accum_l;
    }
    else if (read_and_write[0].isLowRank() && read_only[0].isLowRank())
    {
      // TODO
      printf("Error: Accum dense awaiting implementation.\n");
      inst[0] = (int) nop;
      * tmp_size = 0;
      * rnd_size = 0;
      return nop_l;
    }
    else
    {
      printf("Error: ACCUM on incompatible block.\n");
      inst[0] = (int) nop;
      * tmp_size = 0;
      * rnd_size = 0;
      return nop_l;
    }
  }
  default:
  { 
    inst[0] = (int) nop;
    * tmp_size = 0;
    * rnd_size = 0;
    return nop_l;
  }
  }


}

long long int h_ops::getFlops (long long int * trim)
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
        flops = getFlops_GETRF(trim, nx, ny);
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
        flops = getFlops_TRSML(trim, nx_b, ny_b, nx_l);
      }
      else if (read_and_write[0].isDense() && read_only[0].isDense())
      {
        nx_b = read_and_write[0].getNx();
        ny_b = read_and_write[0].getNy(read_only[0].getNy());
        nx_l = read_only[0].getNx();
        flops = getFlops_TRSML(trim, nx_b, ny_b, nx_l);
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
        flops = getFlops_TRSMR(trim, nx_b, ny_b, ny_u);
      }
      else if (read_and_write[0].isDense() && read_only[0].isDense())
      {
        nx_b = read_and_write[0].getNx(read_only[0].getNx());
        ny_b = read_and_write[0].getNy();
        ny_u = read_only[0].getNy();
        flops = getFlops_TRSMR(trim, nx_b, ny_b, ny_u);
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
      { flops = getFlops_GEMM(trim, m, n, k); break; }
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
      { flops = getFlops_GEMM_3x(trim, m, n, k, l); break; }
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
      { flops = getFlops_GEMM_4x(trim, m, n, k, l, o); }
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

        flops = getFlops_GEMM(trim, m, n, k);
      }
      else if (read_and_write[0].isLowRank() && read_only[0].isLowRank())
      {
        long long int nx = 0, ny = 0, rank1 = 0, rank2 = 0;

        nx = read_and_write[0].getNx(read_only[0].getNx());
        ny = read_and_write[0].getNy(read_only[0].getNy());
        rank1 = read_and_write[0].getRank();
        rank2 = read_only[0].getRank();

        flops = getFlops_LrAccum(trim, nx, ny, rank1, rank2);
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

int h_ops::getTmpSize_ACCM_LR (int * offset1, int * offset2, const int nx, const int ny, const int rank1)
{
  const int size_1 = ny * rank1, size_2 = nx * rank1, size_3 = ny * rank1;
  * offset1 = size_1; * offset2 = size_1 + size_2;
  return size_1 + size_2 + size_3;
}

int h_ops::getControl_GEMM_3x (int * t_size, const int m, const int n, const int k, const int l)
{
  const int size_1 = m * l, size_2 = n * k;

  bool b_ab_a = size_1 * (k + n) <= size_2 * (m + l);
  * t_size = b_ab_a ? size_1 : size_2;

  return (int) b_ab_a; 
}

int h_ops::getControl_GEMM_4x (int * t_size, int * offset, const int m, const int n, const int k, const int l, const int o)
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
    if (b_ab_bc) // ((A x B) x C) x D
    { control = 0; * offset = size_1; * t_size = size_1 + size_2; }
    else // (A x (B x C)) x D
    { control = 1; * offset = size_5; * t_size = size_5 + size_2; }
  }
  else if (b_abc_ab) // A x (B x C x D)
  {
    if (b_bc_cd) // A x ((B x C) x D)
    { control = 2; * offset = size_5; * t_size = size_5 + size_3; }
    else // A x (B x (C x D))
    { control = 3; * offset = size_4; * t_size = size_4 + size_3; }
  }
  else // (A x B) x (C x D)
  { control = 4; * offset = size_1; * t_size = size_1 + size_4; }

  return control; 
}

long long int h_ops::getFlops_GETRF (long long int * trim, const long long int nx, const long long int ny, const long long int trim_dim)
{
  long long int accum = 0, accum_trim = 0;
  long long int n = nx > ny ? ny : nx, nx_trim = nx < trim_dim ? trim_dim : nx, ny_trim = ny < trim_dim ? trim_dim : ny;

  for (long long int i = 0; i < n; i++)
  { 
    accum += (ny - i - 1) * (2 * (nx - i - 1) + 1);
    accum_trim += (ny_trim - i - 1) * (2 * (nx_trim - i - 1) + 1);
  }

  * trim = accum_trim;
  return accum;
}

long long int h_ops::getFlops_TRSML (long long int * trim, const long long int nx_b, const long long int ny_b, const long long int nx_l, const long long int trim_dim)
{
  long long int accum = 0, accum_trim = 0;
  long long int n = nx_l > ny_b ? ny_b : nx_l;
  long long int nx_b_trim = nx_b < trim_dim ? trim_dim : nx_b, ny_b_trim = ny_b < trim_dim ? trim_dim : ny_b;

  accum = (2 * ny_b - n - 1) * n * nx_b;
  accum_trim = (2 * ny_b_trim - n - 1) * n * nx_b_trim;

  * trim = accum_trim;
  return accum;
}

long long int h_ops::getFlops_TRSMR (long long int * trim, const long long int nx_b, const long long int ny_b, const long long int ny_u, const long long int trim_dim)
{
  long long int accum = 0, accum_trim = 0;
  long long int n = nx_b > ny_u ? ny_u : nx_b;
  long long int nx_b_trim = nx_b < trim_dim ? trim_dim : nx_b, ny_b_trim = ny_b < trim_dim ? trim_dim : ny_b;

  accum = (2 * nx_b - n) * n * ny_b;
  accum_trim = (2 * nx_b_trim - n) * n * ny_b_trim;

  * trim = accum_trim;
  return accum;
}

long long int h_ops::getFlops_GEMM (long long int * trim, const long long int m, const long long int n, const long long int k, const long long int trim_dim)
{
  long long int m_trim = m < trim_dim ? trim_dim : m, n_trim = n < trim_dim ? trim_dim : n;
  long long int k_trim = k < trim_dim ? trim_dim : k;
  long long int accum = m * n * k * 2, accum_trim = m_trim * n_trim * k_trim * 2;

  * trim = accum_trim;
  return accum;
}

long long int h_ops::getFlops_GEMM_3x (long long int * trim, const long long int m, const long long int n, const long long int k, const long long int l, const long long int trim_dim)
{
  long long int f1 = k * n * (m + l), f2 = m * l * (k + n), accum = (f1 <= f2 ? f1 : f2) * 2;
  long long int m_trim = m < trim_dim ? trim_dim : m, n_trim = n < trim_dim ? trim_dim : n;
  long long int k_trim = k < trim_dim ? trim_dim : k, l_trim = l < trim_dim ? trim_dim : l;

  f1 = k_trim * n_trim * (m_trim + l_trim); f2 = m_trim * l_trim * (k_trim + n_trim);
  long long int accum_trim = (f1 <= f2 ? f1 : f2) * 2;

  * trim = accum_trim;
  return accum;
}

long long int h_ops::getFlops_GEMM_4x (long long int * trim, const long long int m, const long long int n, const long long int k, const long long int l, const long long int o, const long long int trim_dim)
{
  long long int m_trim = m < trim_dim ? trim_dim : m, n_trim = n < trim_dim ? trim_dim : n;
  long long int k_trim = k < trim_dim ? trim_dim : k, l_trim = l < trim_dim ? trim_dim : l;
  long long int o_trim = o < trim_dim ? trim_dim : o;

  long long int size_1 = m * l, size_2 = m * o, size_3 = n * k, size_4 = n * l, size_5 = k * o;
  long long int size_1_trim = m_trim * l_trim, size_2_trim = m_trim * o_trim, size_3_trim = n_trim * k_trim, size_4_trim = n_trim * l_trim, size_5_trim = k_trim * o_trim;

  long long int f_ab = size_1 * k, f_bc = size_5 * l, f_cd = size_4 * o;
  long long int f_ab_trim = size_1_trim * k_trim, f_bc_trim = size_5_trim * l_trim, f_cd_trim = size_4_trim * o_trim;
  bool b_ab_bc = f_ab <= f_bc, b_bc_cd = f_bc <= f_cd;

  long long int f_abc_d = b_ab_bc ? size_2 * (l + n) + f_ab : size_2 * (k + n) + f_bc;
  long long int f_a_bcd = b_bc_cd ? size_3 * (o + m) + f_bc : size_3 * (l + m) + f_cd;
  long long int f_ab_cd = f_ab + f_cd + size_1 * n;

  long long int f_abc_d_trim = b_ab_bc ? size_2_trim * (l_trim + n_trim) + f_ab_trim : size_2_trim * (k_trim + n_trim) + f_bc_trim;
  long long int f_a_bcd_trim = b_bc_cd ? size_3_trim * (o_trim + m_trim) + f_bc_trim : size_3_trim * (l_trim + m_trim) + f_cd_trim;
  long long int f_ab_cd_trim = f_ab_trim + f_cd_trim + size_1_trim * n_trim;

  long long int f_abcd = f_abc_d <= f_a_bcd ? f_abc_d : f_a_bcd;
  f_abcd = f_abcd <= f_ab_cd ? f_abcd : f_ab_cd;

  long long int f_abcd_trim = f_abc_d_trim <= f_a_bcd_trim ? f_abc_d_trim : f_a_bcd_trim;
  f_abcd_trim = f_abcd_trim <= f_ab_cd_trim ? f_abcd_trim : f_ab_cd_trim;

  * trim = f_abcd_trim * 2;
  return f_abcd * 2;
}

long long int h_ops::getFlops_QR (long long int * trim, const long long int nx, const long long int ny, const long long int trim_dim)
{
  long long int nx_trim = nx < trim_dim ? trim_dim : nx, ny_trim = ny < trim_dim ? trim_dim : ny;
  long long int accum = nx * nx * (3 * ny - nx) * 2, accum_trim = nx_trim * nx_trim * (3 * ny_trim - nx_trim) * 2;

  * trim = accum_trim;
  return accum;
}

long long int h_ops::getFlops_LrAccum (long long int * trim, const long long int nx, const long long int ny, const long long int rank1, const long long int rank2, const long long int trim_dim)
{
  long long int accum = 0, accum_trim = 0, tmp;
  accum += getFlops_GEMM_3x(&tmp, ny, rank1, rank1, nx); accum_trim += tmp;
  accum += getFlops_GEMM_3x(&tmp, ny, rank2, rank1, nx); accum_trim += tmp;
  accum += getFlops_QR(&tmp, rank1, ny); accum_trim += tmp;
  accum += getFlops_GEMM_3x(&tmp, nx, rank1, rank1, ny); accum_trim += tmp;
  accum += getFlops_GEMM_3x(&tmp, nx, rank2, rank1, ny); accum_trim += tmp;

  * trim = accum_trim;
  return accum;
}

void h_ops::print() const
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
  
