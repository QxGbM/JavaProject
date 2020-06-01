
import java.io.*;
import Jama.Matrix;

public class H2Matrix implements Block {

  private ClusterBasis row_basis;
  private ClusterBasis col_basis;
  private Block e[][];
  private int x_start = 0;
  private int y_start = 0;
  private LowRankBasic accm = null;

  private H2Matrix () {
    row_basis = null;
    col_basis = null;
    e = null;
  }

  public H2Matrix (int m, int n, int nleaf, int part_strat, int rank, double admis, int y_start, int x_start, PsplHMatrixPack.DataFunction func) {
    row_basis = new ClusterBasis(y_start, m, true, nleaf, part_strat, rank, admis, func);
    col_basis = new ClusterBasis(x_start, n, false, nleaf, part_strat, rank, admis, func);
    this.x_start = x_start; this.y_start = y_start;
    e = new Block[part_strat][part_strat];

    ClusterBasis[] row_basis_lower = row_basis.getChildren(), col_basis_lower = col_basis.getChildren();

    int m_block = m / part_strat, m_remain = m - (part_strat - 1) * m_block;
    int n_block = n / part_strat, n_remain = n - (part_strat - 1) * n_block;

    for (int i = 0; i < part_strat; i++) {
      int m_e = i == part_strat - 1 ? m_remain : m_block;
      int y_e = y_start + m_block * i;
      ClusterBasis b_i = row_basis_lower[i];

      for (int j = 0; j < part_strat; j++) {
        int n_e = j == part_strat - 1 ? n_remain : n_block;
        int x_e = x_start + n_block * j;
        ClusterBasis b_j = col_basis_lower[j];

        boolean admisible = Integer.max(m_e, n_e) <= admis * Math.abs(x_e - y_e);

        if (admisible)
        { e[i][j] = new Dense(m_e, n_e, y_e, x_e, func).toLowRank_fromBasis(b_i, b_j); }
        else if (b_i.childrenLength() == 0 || b_j.childrenLength() == 0)
        { e[i][j] = new Dense(m_e, n_e, y_e, x_e, func); }
        else
        { e[i][j] = new H2Matrix(b_i, b_j, admis, func); }

      }
    }

    row_basis.convertReducedStorageForm();
    col_basis.convertReducedStorageForm();
  }

  private H2Matrix (ClusterBasis row_basis, ClusterBasis col_basis, double admis, PsplHMatrixPack.DataFunction func) {
    this.row_basis = row_basis;
    this.col_basis = col_basis;
    y_start = row_basis.getStart(); x_start = col_basis.getStart();
    int m = row_basis.getDimension(), n = col_basis.getDimension();
    int part_strat = row_basis.getPartStrat();
    e = new Block[part_strat][part_strat];

    ClusterBasis[] row_basis_lower = row_basis.getChildren(), col_basis_lower = col_basis.getChildren();

    int m_block = m / part_strat, m_remain = m - (part_strat - 1) * m_block;
    int n_block = n / part_strat, n_remain = n - (part_strat - 1) * n_block;

    for (int i = 0; i < part_strat; i++) {
      int m_e = i == part_strat - 1 ? m_remain : m_block;
      int y_e = y_start + m_block * i;
      ClusterBasis b_i = row_basis_lower[i];

      for (int j = 0; j < part_strat; j++) {
        int n_e = j == part_strat - 1 ? n_remain : n_block;
        int x_e = x_start + n_block * j;
        ClusterBasis b_j = col_basis_lower[j];

        boolean admisible = Integer.max(m_e, n_e) <= admis * Math.abs(x_e - y_e);

        if (admisible)
        { e[i][j] = new Dense(m_e, n_e, y_e, x_e, func).toLowRank_fromBasis(b_i, b_j); }
        else if (b_i.childrenLength() == 0 || b_j.childrenLength() == 0)
        { e[i][j] = new Dense(m_e, n_e, y_e, x_e, func); }
        else
        { e[i][j] = new H2Matrix(b_i, b_j, admis, func); }

      }
    }
  }

  public H2Matrix (LowRank lr) {
    row_basis = lr.getU(); 
    col_basis = lr.getVT();
    int m = row_basis.childrenLength() > 0 ? row_basis.childrenLength() : 1;
    int n = col_basis.childrenLength() > 0 ? col_basis.childrenLength() : 1;
    e = new Block[m][n];
    Matrix S = lr.getS();

    for (int i = 0; i < m; i++) {
      Matrix Ss = m > 1 ? row_basis.getTrans(i).times(S) : S;
      ClusterBasis row_i = m > 1 ? row_basis.getChildren()[i] : row_basis;
      for (int j = 0; j < n; j++) {
        Matrix Sss = n > 1 ? Ss.times(col_basis.getTrans(j).transpose()) : Ss;
        ClusterBasis col_j = n > 1 ? col_basis.getChildren()[j] : col_basis;
        e[i][j] = new LowRank(row_i, Sss, col_j);
      }
    }
  }

  public H2Matrix (LowRankBasic lr, int m, int n) {
    row_basis = new ClusterBasis(lr.getU());
    col_basis = new ClusterBasis(lr.getVT());
    e = new Block[m][n];
    row_basis.partition(m);
    col_basis.partition(n);
    int rank = lr.getRank();

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        e[i][j] = new LowRank(row_basis.getChildren()[i], Matrix.identity(rank, rank), col_basis.getChildren()[j]);
      }
    }
  }

  public H2Matrix (Dense d, int m, int n) {
    row_basis = null;
    col_basis = null;
    e = new Block[m][n];
    int y = d.getRowDimension();
    int x = d.getColumnDimension();
    int y_step = y / m;
    int x_step = x / n;

    int y_start = 0;
    for (int i = 0; i < m; i++) {
      int y_end = Integer.min(y_start + y_step, y) - 1;
      int x_start = 0;
      for (int j = 0; j < n; j++) {
        int x_end = Integer.min(x_start + x_step, x) - 1;
        Matrix part = d.getMatrix(y_start, y_end, x_start, x_end);
        e[i][j] = new Dense(part.getArray());
        x_start = x_end + 1;
      }
      y_start = y_end + 1;
    }
  }

  public ClusterBasis getRowBasis() 
  { return row_basis; }

  public ClusterBasis getColBasis()
  { return col_basis; }

  public void setRowBasis (ClusterBasis row_basis) {
    this.row_basis = row_basis;
  }

  public void setColBasis (ClusterBasis col_basis) {
    this.col_basis = col_basis;
  }
  
  public int getNRowBlocks()
  { return e.length; }

  public int getNColumnBlocks()
  { return e[0].length; }

  public Block[][] getElements ()
  { return e; }

  @Override
  public int getXCenter() {
    return x_start + getRowDimension() / 2;
  }

  @Override
  public int getYCenter() {
    return y_start + getColumnDimension() / 2;
  }

  @Override
  public void setClusterStart (int x_start, int y_start) {
    this.x_start = x_start;
    this.y_start = y_start;
  }

  @Override
  public int getRowDimension() {
    int accum = 0;
    for (int i = 0; i < getNRowBlocks(); i++)
    { accum += e[i][0].getRowDimension(); }
    return accum;
  }

  public int getRowDimension(int i) {
    return e[i][0].getRowDimension();
  }

  @Override
  public int getColumnDimension() {
    int accum = 0;
    for (int i = 0; i < getNColumnBlocks(); i++)
    { accum += e[0][i].getColumnDimension(); }
    return accum;
  }

  public int getColumnDimension(int j) {
    return e[0][j].getColumnDimension();
  }

  @Override
  public Block_t getType() 
  { return Block_t.HIERARCHICAL; }

  @Override
  public Dense toDense() {
    Dense d = new Dense(getRowDimension(), getColumnDimension());
    d.setClusterStart(x_start, y_start);
    int i0 = 0;

    for (int i = 0; i < getNRowBlocks(); i++) {
      int i1 = 0, j0 = 0;
      for (int j = 0; j < getNColumnBlocks(); j++) {
        Dense X = e[i][j].toDense(); 
        int j1 = j0 + X.getColumnDimension() - 1;
        i1 = i0 + X.getRowDimension() - 1;
        d.setMatrix(i0, i1, j0, j1, X);
        j0 = j1 + 1;
      }
      i0 = i1 + 1;
    }

    return d;
  }

  @Override
  public LowRank toLowRank() 
  { 
    Matrix approx[][] = new Matrix[getNRowBlocks()][getNColumnBlocks()];

    for (int i = 0; i < getNRowBlocks(); i++) {
      for (int j = 0; j < getNColumnBlocks(); j++) {
        if (e[i][j].getType() == Block_t.DENSE)
        { approx[i][j] = e[i][j].toDense().toLowRank_fromBasis(row_basis.getChildren()[i], col_basis.getChildren()[j]).getS(); }
        else
        { approx[i][j] = e[i][j].toLowRank().getS(); }
      }
    }

    Matrix S = new Matrix (row_basis.getRank(), col_basis.getRank());

    for (int i = 0; i < getNRowBlocks(); i++) {
      for (int j = 0; j < getNColumnBlocks(); j++) {
        Matrix Ss = row_basis.getTrans(i).transpose().times(approx[i][j]).times(col_basis.getTrans(j));
        S.plusEquals(Ss);
      }
    }

    return new LowRank(row_basis, S, col_basis);
  }

  @Override
  public LowRankBasic toLowRankBasic () { 
    return new LowRankBasic(toLowRank());
  }

  @Override
  public Hierarchical castHierarchical() {
    return null;
  }

  @Override
  public H2Matrix castH2Matrix() {
    return this;
  }

  @Override
  public void setAccumulator (LowRankBasic accm) {
    this.accm = accm;
  }

  @Override
  public LowRankBasic getAccumulator() {
    return accm;
  }

  @Override
  public double compare (Matrix m) {
    return this.toDense().minus(m).normF() / getColumnDimension() / getRowDimension();
  }

  @Override
  public double getCompressionRatio () {
    double compress = 0.;

    for (int i = 0; i < getNRowBlocks(); i++) {
      for (int j = 0; j < getNColumnBlocks(); j++)
      { compress += e[i][j].getCompressionRatio_NoBasis() * e[i][j].getRowDimension() * e[i][j].getColumnDimension(); }
    }

    compress += row_basis.size() + col_basis.size();
    return compress / getColumnDimension() / getRowDimension();
  }

  @Override
  public double getCompressionRatio_NoBasis () {
    double compress = 0.;

    for (int i = 0; i < getNRowBlocks(); i++) {
      for (int j = 0; j < getNColumnBlocks(); j++)
      { compress += e[i][j].getCompressionRatio_NoBasis() * e[i][j].getRowDimension() * e[i][j].getColumnDimension(); }
    }

    return compress / getColumnDimension() / getRowDimension();
  }

  @Override
  public String structure () {
    String s = "H " + Integer.toString(getNRowBlocks()) + " " + Integer.toString(getNColumnBlocks()) + "\n";

    for (int i = 0; i < getNRowBlocks(); i++) {
      for (int j = 0; j < getNColumnBlocks(); j++)
      { s += e[i][j].structure(); }
    }

    return s;
  }

  @Override
  public void loadBinary (InputStream stream) throws IOException {
    for (int i = 0; i < getNRowBlocks(); i++) {
      for (int j = 0; j < getNColumnBlocks(); j++)
      { e[i][j].loadBinary(stream); }
    }
  }

  @Override
  public void writeBinary (OutputStream stream) throws IOException {
    for (int i = 0; i < getNRowBlocks(); i++) {
      for (int j = 0; j < getNColumnBlocks(); j++)
      { e[i][j].writeBinary(stream); }
    }
  }


  @Override
  public void print (int w, int d) {
    for (int i = 0; i < getNRowBlocks(); i++) {
      for (int j = 0; j < getNColumnBlocks(); j++)
      { e[i][j].print(w, d); }
    }
  }

  @Override
  public Block LU () {
    int m = getNRowBlocks(), n = getNColumnBlocks(), iters = Integer.min(m, n);
    if (getAccumulator() != null)
    { accum(accm); }

    for (int i = 0; i < iters; i++) {
      e[i][i].LU();
      for (int j = i + 1; j < m; j++) {
        e[j][i].triangularSolve(e[i][i], true);
      }
      for (int j = i + 1; j < n; j++) {
        e[i][j].triangularSolve(e[i][i], false);
        for (int k = i + 1; k < m; k++) {
          e[k][j].GEMatrixMult(e[k][i], e[i][j], -1., 1.);
        }
      }
    }

    return this;
  }

  @Override
  public Block triangularSolve (Block b, boolean up_low) {
    if (getAccumulator() != null)
    { accum(accm); }

    if (b.castH2Matrix() != null)
    { return triangularSolve(b.castH2Matrix(), up_low); }
    else
    { return this; }
  }

  public H2Matrix triangularSolve (H2Matrix h, boolean up_low) {

    int m = getNRowBlocks(), n = getNColumnBlocks();
    if (up_low) {
      for (int j = 0; j < n; j++) {
        for (int i = 0; i < m; i++) {
          e[i][j].triangularSolve(h.e[j][j], up_low);
          for (int k = j + 1; k < n; k++) 
          { e[i][k].GEMatrixMult(e[i][j], h.e[j][k], -1., 1.); }
        }
      }
    }
    else {
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          e[i][j].triangularSolve(h.e[i][i], up_low);
          for (int k = i + 1; k < m; k++) 
          { e[k][j].GEMatrixMult(h.e[k][i], e[i][j], -1., 1.); }
        }
      }
    }

    return this;
  }

  @Override
  public Block GEMatrixMult (Block a, Block b, double alpha, double beta) {
    scalarEquals(beta);
    Block c = a.times(b);
    c.scalarEquals(alpha);
    plusEquals(c);
    return this;
  }

  @Override
  public Block GEMatrixMult (Block a, Block b, double alpha, double beta, ClusterBasisProduct X, ClusterBasisProduct Y, ClusterBasisProduct Z, H2Approx Sa, H2Approx Sb, H2Approx Sc) {
    if (a.getType() == Block_t.LOW_RANK) {
      if (b.getType() == Block_t.LOW_RANK) 
      { GEMatrixMult(a.toLowRank(), b.toLowRank(), alpha, beta, X, Y, Z, Sa, Sb, Sc); }
      else {
        Sa.splitProduct(row_basis, b.castH2Matrix().row_basis);
        LowRank a_prime = new LowRank (row_basis, b.castH2Matrix().row_basis, a.toLowRank());
        GEMatrixMult(new H2Matrix(a_prime), b.castH2Matrix(), alpha, beta, X, Y, Z, Sa, Sb, Sc); 
      }
    }
    else {
      if (b.getType() == Block_t.LOW_RANK) {
        Sb.splitProduct(a.castH2Matrix().col_basis, col_basis);
        LowRank b_prime = new LowRank (a.castH2Matrix().col_basis, col_basis, b.toLowRank());
        GEMatrixMult(a.castH2Matrix(), new H2Matrix(b_prime), alpha, beta, X, Y, Z, Sa, Sb, Sc); 
      }
      else
      { GEMatrixMult(a.castH2Matrix(), b.castH2Matrix(), alpha, beta, X, Y, Z, Sa, Sb, Sc); }
    }
    return this;
  }

  public H2Matrix GEMatrixMult (H2Matrix a, H2Matrix b, double alpha, double beta) {
    ClusterBasisProduct X = new ClusterBasisProduct(row_basis, a.row_basis);
    ClusterBasisProduct Y = new ClusterBasisProduct(a.col_basis, b.row_basis);
    ClusterBasisProduct Z = new ClusterBasisProduct(b.col_basis, col_basis);

    H2Approx Sa = new H2Approx(row_basis, b.row_basis, X, Y, a);
    H2Approx Sb = new H2Approx(a.col_basis, col_basis, Y, Z, b);
    H2Approx Sc = new H2Approx(getNRowBlocks(), getNColumnBlocks());

    GEMatrixMult (a, b, alpha, beta, X, Y, Z, Sa, Sb, Sc);
    matrixBack(a.row_basis, b.col_basis, X, Z, Sc);
    return this;
  }

  public Block GEMatrixMult (LowRank a, LowRank b, double alpha, double beta, ClusterBasisProduct X, ClusterBasisProduct Y, ClusterBasisProduct Z, H2Approx Sa, H2Approx Sb, H2Approx Sc) {
    scalarEquals(beta);
    Matrix m = Sa.getS().times(Y.getProduct()).times(Sb.getS()).times(alpha);
    Sc.accumProduct(m);
    return this;
  }
  
  public H2Matrix GEMatrixMult (H2Matrix a, H2Matrix b, double alpha, double beta, ClusterBasisProduct X, ClusterBasisProduct Y, ClusterBasisProduct Z, H2Approx Sa, H2Approx Sb, H2Approx Sc) {
    int m = getNRowBlocks(), n = getNColumnBlocks(), k = a.getNColumnBlocks();

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        if (beta != 1.) 
        { e[i][j].scalarEquals(beta); }
        H2Matrix h_ij = e[i][j].castH2Matrix();
        H2Approx Sc_prime = h_ij != null ? Sc.expandChildren(i, j, h_ij.getNRowBlocks(), h_ij.getNColumnBlocks()) : Sc.getChildren(i, j);

        for (int kk = 0; kk < k; kk++) 
        { e[i][j].GEMatrixMult(a.e[i][kk], b.e[kk][j], alpha, 1., X.getChildren(i), Y.getChildren(kk), Z.getChildren(j), Sa.getChildren(i, kk), Sb.getChildren(kk, j), Sc_prime); }
      }
    }

    return this;
  }

  public H2Matrix matrixBack (ClusterBasis left_prime, ClusterBasis right_prime, ClusterBasisProduct X, ClusterBasisProduct Y, H2Approx S) {
    int m = getNRowBlocks(), n = getNColumnBlocks();
    S.splitProduct(left_prime, right_prime);

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        if (S.hasChildren(i, j)) {
          Block b = getElement(i, j);
          if (b.getType() == Block.Block_t.LOW_RANK) 
          { b.toLowRank().plusEquals(X.getChildren(i), Y.getChildren(j), S.getProduct(i, j)); } 
          else if (b.getType() == Block.Block_t.DENSE) 
          { b.toDense().plusEquals(left_prime.childMatrix(i).times(S.getProduct(i, j)).times(right_prime.childMatrix(j).transpose())); }
          else 
          { b.castH2Matrix().matrixBack(left_prime.getChildren()[i], right_prime.getChildren()[j], X.getChildren(i), Y.getChildren(j), S.getChildren(i, j)); }
        }
      }
    }

    return this;
  }

  public H2Matrix plusEquals (H2Matrix h, H2Approx Sa, H2Approx Sb) {
    for (int i = 0; i < getNRowBlocks(); i++) {
      for (int j = 0; j < getNColumnBlocks(); j++) {

        if (h.e[i][j].getType() == Block_t.LOW_RANK || h.e[i][j].getType() == Block_t.DENSE) {
          if (h.e[i][j].getType() == Block_t.LOW_RANK)
          { Sb.accumProduct(i, j, h.e[i][j].toLowRank().getS()); }
          else if (e[i][j].getType() == Block_t.LOW_RANK)
          { e[i][j].toLowRank().getS().plusEquals(Sa.getProduct(i, j)); }
          else
          { e[i][j] = e[i][j].toDense().plusEquals(h.e[i][j].toDense()); }
        }
        else if (e[i][j].getType() == Block_t.LOW_RANK)
        { e[i][j].toLowRank().getS().plusEquals(Sa.getProduct(i, j)); }
        else if (e[i][j].getType() == Block_t.DENSE) 
        { e[i][j].toDense().plusEquals(h.e[i][j].toDense()); }
        else {
          H2Matrix h_e = e[i][j].castH2Matrix();
          H2Approx Sb_prime = Sb.expandChildren(i, j, h_e.getNRowBlocks(), h_e.getNColumnBlocks());
          h_e.plusEquals(h.e[i][j].castH2Matrix(), Sa.getChildren(i, j), Sb_prime);
        }
        
      }
    }
    return this;
  }

  @Override
  public Block plusEquals (Block b) {
    if (b.castH2Matrix() != null)
    { return plusEquals(b.castH2Matrix()); }
    else if (b.getType() == Block_t.DENSE) {
      H2Matrix h = new H2Matrix(b.toDense(), getNRowBlocks(), getNColumnBlocks());
      return plusEquals(h);
    }
    else if (b.getType() == Block_t.LOW_RANK) {
      LowRankBasic lr = b.toLowRankBasic();
      if (accm == null)
      { accm = new LowRankBasic(); }
      accm.plusEquals(lr.toLowRankBasic());
      return this;
    }
    else
    { return null; }
  }

  public H2Matrix plusEquals (H2Matrix h) {
    for (int i = 0; i < getNRowBlocks(); i++) {
      for (int j = 0; j < getNColumnBlocks(); j++) {
        if (h.e[i][j].getAccumulator() != null)
        { e[i][j].plusEquals(h.e[i][j].getAccumulator()); }
        e[i][j].plusEquals(h.e[i][j]);
      }
    }
    return this;
  }

  @Override
  public Block scalarEquals (double s) {
    for (int i = 0; i < getNRowBlocks(); i++) {
      for (int j = 0; j < getNColumnBlocks(); j++) {
        e[i][j].scalarEquals(s);
      }
    }
    return this;
  }

  @Override
  public Block times (Block b) {
    if (b.castH2Matrix() != null)
    { return times(b.castH2Matrix()); }
    else if (b.getType() == Block_t.DENSE) 
    { return null; }
    else if (b.getType() == Block_t.LOW_RANK) 
    { return times(b.toLowRank()); }
    else 
    { return null; }
  }


  public H2Matrix times (H2Matrix h) {
    int l = getNColumnBlocks();
    if (l == h.getNRowBlocks()) { 
      H2Matrix product = new H2Matrix();
      product.row_basis = row_basis;
      product.col_basis = h.col_basis;
      int m = getNRowBlocks();
      int n = h.getNColumnBlocks();
      product.e = new Block[m][n];
  
      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          product.e[i][j] = e[i][0].times(h.e[0][j]);
          for (int k = 1; k < l; k++) {
            Block b = e[i][k].times(h.e[k][j]);
            if (product.e[i][j].getType() == Block_t.LOW_RANK && b.castH2Matrix() != null)
            { b.plusEquals(product.e[i][j]); product.e[i][j] = b; }
            else
            { product.e[i][j].plusEquals(b); }
          }
        }
      }
      return product;
    }
    return null;
  }

  public LowRankBasic times (LowRank lr) {
    int rank = lr.getRank();
    Matrix u = lr.getU().h2matrixTimes(this, false);
    Matrix us = u.times(lr.getS());
    Matrix vt = lr.getVT().toMatrix(rank);
    return new LowRankBasic(us, vt);
  }

  @Override
  public Block accum (LowRankBasic accm) {
    return plusEquals(new H2Matrix(accm, getNRowBlocks(), getNColumnBlocks()));
  }

  public Block getElement (int m, int n)
  { return e[m][n]; }

  public void setElement (int m, int n, Block b) {
    if (m < getNRowBlocks() && n < getNColumnBlocks())
    { e[m][n] = b; }
  }

  public void compareDense (Matrix d, String str) {
    int m = getNRowBlocks(), n = getNColumnBlocks();
    int y = 0;
    for (int i = 0; i < m; i++) {
      int y_end = y + e[i][0].getRowDimension() - 1, x = 0;
      for (int j = 0; j < n; j++) {
        int x_end = x + e[i][j].getColumnDimension() - 1;
        double norm = e[i][j].compare(d.getMatrix(y, y_end, x, x_end));
        System.out.println(str + " (" + i + ", " + j + "): " + norm);
        if (e[i][j].castH2Matrix() != null)
        { e[i][j].castH2Matrix().compareDense(d.getMatrix(y, y_end, x, x_end), str + " (" + i + ", " + j + ")"); }
        x = x_end + 1;
      }
      y = y_end + 1;
    }

  }



}