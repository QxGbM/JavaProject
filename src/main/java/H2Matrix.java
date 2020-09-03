
import java.io.*;
import Jama.Matrix;

public class H2Matrix implements Block {

  private ClusterBasis rowBasis;
  private ClusterBasis colBasis;
  private Block[][] e;
  private LowRankBasic accm = null;

  private H2Matrix () {
    rowBasis = null;
    colBasis = null;
    e = null;
  }

  public H2Matrix (ClusterBasis rowBasis, ClusterBasis colBasis, int yStart, int xStart, double admis, PsplHMatrixPack.DataFunction func) {
    this.rowBasis = rowBasis;
    this.colBasis = colBasis;
    int m = rowBasis.getDimension();
    int n = colBasis.getDimension();
    int partStrat = rowBasis.getPartStrat();
    e = new Block[partStrat][partStrat];

    ClusterBasis[] rowBasisLower = rowBasis.getChildren();
    ClusterBasis[] colBasisLower = colBasis.getChildren();

    int mBlock = m / partStrat;
    int mRemain = m - (partStrat - 1) * mBlock;
    int nBlock = n / partStrat;
    int nRemain = n - (partStrat - 1) * nBlock;

    for (int i = 0; i < partStrat; i++) {
      int mE = i == partStrat - 1 ? mRemain : mBlock;
      int yE = yStart + mBlock * i;
      ClusterBasis bI = rowBasisLower[i];

      for (int j = 0; j < partStrat; j++) {
        int nE = j == partStrat - 1 ? nRemain : nBlock;
        int xE = xStart + nBlock * j;
        ClusterBasis bJ = colBasisLower[j];

        boolean admisible = Integer.max(mE, nE) <= admis * Math.abs(xE - yE);

        if (admisible)
        { e[i][j] = new Dense(mE, nE, yE, xE, func).toLowRankFromBasis(bI, bJ); }
        else if (bI.childrenLength() == 0 || bJ.childrenLength() == 0)
        { e[i][j] = new Dense(mE, nE, yE, xE, func); }
        else
        { e[i][j] = new H2Matrix(bI, bJ, yE, xE, admis, func); }

      }
    }
  }

  public H2Matrix (LowRank lr) {
    rowBasis = lr.getU(); 
    colBasis = lr.getVT();
    int m = rowBasis.childrenLength() > 0 ? rowBasis.childrenLength() : 1;
    int n = colBasis.childrenLength() > 0 ? colBasis.childrenLength() : 1;
    e = new Block[m][n];
    Matrix s = lr.getS();

    for (int i = 0; i < m; i++) {
      Matrix ss = m > 1 ? rowBasis.getTrans(i).times(s) : s;
      ClusterBasis rowI = m > 1 ? rowBasis.getChildren()[i] : rowBasis;
      for (int j = 0; j < n; j++) {
        Matrix sss = n > 1 ? ss.times(colBasis.getTrans(j).transpose()) : ss;
        ClusterBasis colJ = n > 1 ? colBasis.getChildren()[j] : colBasis;
        e[i][j] = new LowRank(rowI, sss, colJ);
      }
    }
  }

  public H2Matrix (LowRankBasic lr, int m, int n) {
    rowBasis = new ClusterBasis(lr.getU());
    colBasis = new ClusterBasis(lr.getVT());
    e = new Block[m][n];
    rowBasis.partition(m);
    colBasis.partition(n);
    int rank = lr.getRank();

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        e[i][j] = new LowRank(rowBasis.getChildren()[i], Matrix.identity(rank, rank), colBasis.getChildren()[j]);
      }
    }
  }

  public H2Matrix (Dense d, int m, int n) {
    rowBasis = null;
    colBasis = null;
    e = new Block[m][n];
    int y = d.getRowDimension();
    int x = d.getColumnDimension();
    int yStep = y / m;
    int xStep = x / n;

    int yStart = 0;
    for (int i = 0; i < m; i++) {
      int yEnd = Integer.min(yStart + yStep, y) - 1;
      int xStart = 0;
      for (int j = 0; j < n; j++) {
        int xEnd = Integer.min(xStart + xStep, x) - 1;
        Matrix part = d.getMatrix(yStart, yEnd, xStart, xEnd);
        e[i][j] = new Dense(part.getArray());
        xStart = xEnd + 1;
      }
      yStart = yEnd + 1;
    }
  }

  public ClusterBasis getRowBasis() 
  { return rowBasis; }

  public ClusterBasis getColBasis()
  { return colBasis; }

  public void setRowBasis (ClusterBasis rowBasis) {
    this.rowBasis = rowBasis;
  }

  public void setColBasis (ClusterBasis colBasis) {
    this.colBasis = colBasis;
  }
  
  public int getNRowBlocks()
  { return e.length; }

  public int getNColumnBlocks()
  { return e[0].length; }

  public Block[][] getElements ()
  { return e; }

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
    int i0 = 0;

    for (int i = 0; i < getNRowBlocks(); i++) {
      int i1 = 0;
      int j0 = 0;
      for (int j = 0; j < getNColumnBlocks(); j++) {
        Dense x = e[i][j].toDense(); 
        int j1 = j0 + x.getColumnDimension() - 1;
        i1 = i0 + x.getRowDimension() - 1;
        d.setMatrix(i0, i1, j0, j1, x);
        j0 = j1 + 1;
      }
      i0 = i1 + 1;
    }

    return d;
  }

  @Override
  public LowRank toLowRank() 
  { 
    Matrix[][] approx = new Matrix[getNRowBlocks()][getNColumnBlocks()];

    for (int i = 0; i < getNRowBlocks(); i++) {
      ClusterBasis rb = rowBasis.getChildren()[i];
      for (int j = 0; j < getNColumnBlocks(); j++) {
        ClusterBasis cb = colBasis.getChildren()[j];
        if (e[i][j].getType() == Block_t.LOW_RANK)
        { approx[i][j] = ClusterBasisProduct.alignRank(e[i][j].toLowRank().getS(), rb.getRank(), cb.getRank()); }
        else
        { approx[i][j] = e[i][j].toDense().toLowRankFromBasis(rb, cb).getS(); }
      }
    }

    Matrix s = new Matrix (rowBasis.getRank(), colBasis.getRank());

    for (int i = 0; i < getNRowBlocks(); i++) {
      for (int j = 0; j < getNColumnBlocks(); j++) {
        Matrix ss = rowBasis.getTrans(i).transpose().times(approx[i][j]).times(colBasis.getTrans(j));
        s.plusEquals(ss);
      }
    }

    return new LowRank(rowBasis, s, colBasis);
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
    double norm = 0.;
    int i0 = 0;
    for (int i = 0; i < getNRowBlocks(); i++) {
      int i1 = 0;
      int j0 = 0;
      for (int j = 0; j < getNColumnBlocks(); j++) {
        int l = e[i][j].getRowDimension();
        int n = e[i][j].getColumnDimension();
        int j1 = j0 + n - 1;
        i1 = i0 + l - 1;
        norm += e[i][j].compare(m.getMatrix(i0, i1, j0, j1));
        j0 = j1 + 1;
      }
      i0 = i1 + 1;
    }
    return norm;
  }

  @Override
  public double getCompressionRatio () {
    double compress = 0.;

    for (int i = 0; i < getNRowBlocks(); i++) {
      for (int j = 0; j < getNColumnBlocks(); j++)
      { compress += e[i][j].getCompressionRatioNoBasis() * e[i][j].getRowDimension() * e[i][j].getColumnDimension(); }
    }

    compress += rowBasis.size() + colBasis.size();
    return compress / getColumnDimension() / getRowDimension();
  }

  @Override
  public double getCompressionRatioNoBasis () {
    double compress = 0.;

    for (int i = 0; i < getNRowBlocks(); i++) {
      for (int j = 0; j < getNColumnBlocks(); j++)
      { compress += e[i][j].getCompressionRatioNoBasis() * e[i][j].getRowDimension() * e[i][j].getColumnDimension(); }
    }

    return compress / getColumnDimension() / getRowDimension();
  }

  @Override
  public String structure () {
    StringBuilder s = new StringBuilder("H " + Integer.toString(getNRowBlocks()) + " " + Integer.toString(getNColumnBlocks()) + "\n");

    for (int i = 0; i < getNRowBlocks(); i++) {
      for (int j = 0; j < getNColumnBlocks(); j++)
      { s.append(e[i][j].structure()); }
    }

    return s.toString();
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
  public Block getrf () {
    int m = getNRowBlocks();
    int n = getNColumnBlocks();
    int iters = Integer.min(m, n);
    if (getAccumulator() != null)
    { accum(accm); }

    for (int i = 0; i < iters; i++) {
      e[i][i].getrf();
      for (int j = i + 1; j < m; j++) {
        e[j][i].trsm(e[i][i], false);
      }
      for (int j = i + 1; j < n; j++) {
        e[i][j].trsm(e[i][i], true);
        for (int k = i + 1; k < m; k++) {
          e[k][j].gemm(e[k][i], e[i][j], -1., 1.);
        }
      }
    }

    return this;
  }

  @Override
  public Block trsm (Block b, boolean lower) {
    if (getAccumulator() != null)
    { accum(accm); }

    if (b.castH2Matrix() != null)
    { return lower ? trsml(b.castH2Matrix()) : trsmr(b.castH2Matrix()); }
    else
    { return this; }
  }

  public H2Matrix trsml (H2Matrix h) {
    int m = getNRowBlocks();
    int n = getNColumnBlocks();
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        e[i][j].trsm(h.e[i][i], true);
        for (int k = i + 1; k < m; k++) 
        { e[k][j].gemm(h.e[k][i], e[i][j], -1., 1.); }
      }
    }
    return this;
  }

  public H2Matrix trsmr (H2Matrix h) {
    int m = getNRowBlocks();
    int n = getNColumnBlocks();
    for (int j = 0; j < n; j++) {
      for (int i = 0; i < m; i++) {
        e[i][j].trsm(h.e[j][j], false);
        for (int k = j + 1; k < n; k++) 
        { e[i][k].gemm(e[i][j], h.e[j][k], -1., 1.); }
      }
    }
    return this;
  }


  @Override
  public Block gemm (Block a, Block b, double alpha, double beta) {
    scalarEquals(beta);
    Block c = a.times(b);
    c.scalarEquals(alpha);
    plusEquals(c);
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

  public H2Matrix plusEqualsDeep (H2Matrix h) {
    for (int i = 0; i < getNRowBlocks(); i++) {
      for (int j = 0; j < getNColumnBlocks(); j++) {
        if (h.e[i][j].getAccumulator() != null)
        { e[i][j].plusEquals(h.e[i][j].getAccumulator()); }
        if (h.e[i][j].castH2Matrix() != null && e[i][j].getType() == Block_t.LOW_RANK) { 
          H2Matrix b = h.e[i][j].copyBlock().castH2Matrix();
          e[i][j] = b.plusEquals(e[i][j].toLowRankBasic());
        }
        else if (h.e[i][j].castH2Matrix() != null && e[i][j].castH2Matrix() != null)
        { e[i][j].castH2Matrix().plusEqualsDeep(h.e[i][j].castH2Matrix()); }
        else
        { e[i][j].plusEquals(h.e[i][j]); }
      }
    }
    return this;
  }

  @Override
  public Block scalarEquals (double s) {
    if (accm != null)
    { accm.scalarEquals(s); }
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
    if (l != h.getNRowBlocks()) 
    { PsplHMatrixPack.errorOut("error partition"); return null; }

    H2Matrix product = new H2Matrix();
    product.rowBasis = rowBasis;
    product.colBasis = h.colBasis;
    int m = getNRowBlocks();
    int n = h.getNColumnBlocks();
    product.e = new Block[m][n];

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        product.e[i][j] = e[i][0].times(h.e[0][j]);
        for (int k = 1; k < l; k++) {
          Block b = e[i][k].times(h.e[k][j]);
          if (product.e[i][j].getType() == Block_t.LOW_RANK && b.castH2Matrix() != null)
          { product.e[i][j] = b.plusEquals(product.e[i][j]); }
          else if (product.e[i][j].castH2Matrix() != null && b.castH2Matrix() != null)
          { product.e[i][j].castH2Matrix().plusEqualsDeep(b.castH2Matrix()); }
          else
          { product.e[i][j].plusEquals(b); }
        }
      }
    }
    return product;
  }

  public LowRankBasic times (LowRank lr) {
    int rank = lr.getRank();
    Matrix u = lr.getU().h2matrixTimes(this, false);
    Matrix us = u.getMatrix(0, lr.getRowDimension() - 1, 0, rank - 1).times(lr.getS());
    Matrix vt = lr.getVT().toMatrix(rank);
    return new LowRankBasic(us, vt);
  }

  public Matrix times (Matrix vec, boolean transpose) {
    ClusterBasis col = transpose ? getRowBasis() : getColBasis();
    ClusterBasisProduct forward = new ClusterBasisProduct(col, vec);

    ClusterBasis row = transpose ? getColBasis() : getRowBasis();
    ClusterBasisProduct accmAdmis = new ClusterBasisProduct();
    Matrix accmY = ClusterBasisProduct.basisInteract(vec, this, forward, accmAdmis, row, transpose);
    
    if (accmY == null)
    { accmY = new Matrix(row.getDimension(), vec.getColumnDimension()); }
    accmY = accmAdmis.accmAdmisBackward(row, accmY);
    return accmY;
  }

  @Override
  public Block accum (LowRankBasic accm) {
    return plusEquals(new H2Matrix(accm, getNRowBlocks(), getNColumnBlocks()));
  }

  @Override
  public Block copyBlock () {
    H2Matrix h = new H2Matrix();
    h.rowBasis = rowBasis;
    h.colBasis = colBasis;
    int m = getNRowBlocks();
    int n = getNColumnBlocks();
    h.e = new Block[m][n];
    if (accm != null)
    { h.setAccumulator(accm.copyBlock().toLowRankBasic()); }
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        h.e[i][j] = e[i][j].copyBlock();
      }
    }
    return h;
  }

  public Block getElement (int m, int n)
  { return e[m][n]; }

  public void setElement (int m, int n, Block b) {
    if (m < getNRowBlocks() && n < getNColumnBlocks())
    { e[m][n] = b; }
  }

  public String compareDense (Matrix d, String str) {
    int m = getNRowBlocks();
    int n = getNColumnBlocks();
    int y = 0;
    StringBuilder strB = new StringBuilder();
    for (int i = 0; i < m; i++) {
      int yEnd = y + e[i][0].getRowDimension() - 1;
      int x = 0;
      for (int j = 0; j < n; j++) {
        int xEnd = x + e[i][j].getColumnDimension() - 1;
        double norm = e[i][j].compare(d.getMatrix(y, yEnd, x, xEnd));
        strB.append(str + "(" + i + ", " + j + "): " + norm + "\n");
        if (e[i][j].castH2Matrix() != null)
        { strB.append(e[i][j].castH2Matrix().compareDense(d.getMatrix(y, yEnd, x, xEnd), str + "(" + i + ", " + j + ") ")); }
        x = xEnd + 1;
      }
      y = yEnd + 1;
    }
    return strB.toString();

  }



}