
import java.io.*;
import java.nio.ByteBuffer;

import Jama.LUDecomposition;
import Jama.Matrix;
import Jama.QRDecomposition;
import Jama.SingularValueDecomposition;

public class Dense extends Matrix implements Block 
{
  private static final long serialVersionUID = 1;
  private transient LowRankBasic accm = null;

  public Dense (double[][] a)
  { super(a); }

  public Dense (double[][] a, int m, int n) 
  { super(a, m, n); }

  public Dense (double[] vals, int m) 
  { super(vals, m); }

  public Dense (int m, int n)
  { super(m, n); }

  public Dense (int m, int n, int yStart, int xStart, PsplHMatrixPack.DataFunction func)
  {
    super(m, n);
    double[][] data = getArray();

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++)
      { data[i][j] = func.body(yStart + i, xStart + j); }
    }

  }

  public Dense (int m, int n, double s)
  { super(m, n, s); }

  @Override
  public Block_t getType()
  { return Block_t.DENSE; }

  @Override
  public Dense toDense()
  { return this; }

  @Override
  public LowRank toLowRank() {
    int m = getRowDimension();
    int n = getColumnDimension();
    int step = n < 4 ? 1 : n / 4;
    int r = 0;
    step = n > PsplHMatrixPack.rank * 4 ? PsplHMatrixPack.rank : step;

    boolean approx;
    Matrix q;
    Matrix y;
    QRDecomposition qrd;
    
    do {
      r += step; r = r >= n ? n : r;

      Matrix x = times(random(n, r));
      qrd = x.qr();
      q = qrd.getQ();
      y = transpose().times(q);
      Matrix a = q.times(y.transpose()).minus(this);
      double norm = a.normF() / (m * n);
      approx = norm <= PsplHMatrixPack.EPI;

    } while (r < n && !approx);

    qrd = y.qr();
    Matrix v = qrd.getQ();
    Matrix u = qrd.getR();
    SingularValueDecomposition svdd = u.svd();

    ClusterBasis rowB = new ClusterBasis(q.times(svdd.getV()));
    ClusterBasis colB = new ClusterBasis(v.times(svdd.getU()));

    return new LowRank (rowB, svdd.getS(), colB);
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
    return null;
  }

  public LowRank toLowRankFromBasis (ClusterBasis row_basis, ClusterBasis col_basis_t) {
    Matrix S = row_basis.toMatrix().transpose().times(this).times(col_basis_t.toMatrix());
    return new LowRank(row_basis, S, col_basis_t);
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
  public Dense transpose () {
    return new Dense(super.transpose().getArray());
  }

  @Override
  public double compare (Matrix m) {
    return this.minus(m).normF() / getColumnDimension() / getRowDimension();
  }

  @Override
  public double getCompressionRatio () 
  { return 1.; }

  @Override
  public double getCompressionRatioNoBasis () 
  { return 1.; }

  @Override
  public String structure ()
  { return "D " + Integer.toString(getRowDimension()) + " " + Integer.toString(getColumnDimension()) + "\n"; }

  @Override
  public void loadBinary (InputStream stream) throws IOException
  {
    int m = getRowDimension();
    int n = getColumnDimension();
    byte[] data;
    double[][] dataPtr = getArray();

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        data = stream.readNBytes(8 * m * n);
        dataPtr[i][j] = ByteBuffer.wrap(data).getDouble(0); 
      }
    }
  }

  @Override
  public void writeBinary (OutputStream stream) throws IOException {
    int m = getRowDimension();
    int n = getColumnDimension();
    byte[] data = new byte[8];
    double[][] data_ptr = getArray();

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) { 
        ByteBuffer.wrap(data).putDouble(0, data_ptr[i][j]);
        stream.write(data);
      }
    }

  }

  @Override
  public Block getrf () {
    if (getAccumulator() != null)
    { accum(accm); }
    LUDecomposition lu_ = lu();
    Matrix L = lu_.getL();
    Matrix U = lu_.getU();
    for (int i = 0; i < getRowDimension(); i++) {
      for (int j = 0; j < getColumnDimension(); j++) {
        set(i, j, i > j ? L.get(i, j) : U.get(i, j));
      }
    }
    return this;
  }

  public Matrix getL() {
    Matrix L = new Matrix(getArrayCopy());
    for (int i = 0; i < getRowDimension(); i++) {
      L.set(i, i, 1);
      for (int j = i + 1; j < getColumnDimension(); j++) 
      { L.set(i, j, 0); }
    }
    return L;
  }

  public Matrix getU() {
    Matrix U = new Matrix(getArrayCopy());
    for (int i = 0; i < getRowDimension(); i++) {
      for (int j = 0; j < i; j++) 
      { U.set(i, j, 0); }
    }
    return U;
  }

  @Override
  public Block trsm (Block b, boolean lower) {
    if (getAccumulator() != null)
    { accum(accm); }
    return trsm(b.toDense(), lower);
  }

  public Dense trsm (Dense d, boolean lower) {
    Matrix m = lower ? d.getU().solveTranspose(this).transpose() : d.getL().solve(this);
    setMatrix(0, getRowDimension() - 1, 0, getColumnDimension() - 1, m);
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
    if (b.getType() == Block_t.DENSE)
    { return plusEquals(b.toDense()); }
    else if (b.getType() == Block_t.LOW_RANK) {
      LowRankBasic lr = b.toLowRankBasic();
      if (accm == null)
      { accm = new LowRankBasic(); }
      accm.plusEquals(lr.toLowRankBasic());
      return this;
    }
    else if (b.getType() == Block_t.HIERARCHICAL) 
    { return plusEquals(b.toDense()); }
    else
    { return this; }
  }

  public Dense plusEquals (Dense d) {
    super.plusEquals(d);
    return this;
  }

  @Override
  public Block scalarEquals (double s) {
    if (s != 1.)
    { super.timesEquals(s); }
    return this;
  }

  @Override
  public Block times (Block b) {
    if (b.getType() == Block_t.LOW_RANK)
    { return times(b.toLowRank()); }
    else
    { return times(b.toDense()); }
  }

  public Dense times (Dense d) {
    Matrix R = super.times(d);
    return new Dense (R.getArrayCopy());
  }

  public LowRankBasic times (LowRank lr) {
    int rank = lr.getS().getColumnDimension();
    Matrix u = times(lr.getUS());
    Matrix vt = lr.getVT().toMatrix(rank);
    return new LowRankBasic(u, vt);
  }

  @Override
  public Block accum (LowRankBasic accm) {
    return plusEquals(accm.toDense());
  }

  public static Matrix getBasisU (int yStart, int m, int rank, double admis, PsplHMatrixPack.DataFunction func) {
    int minimal_sep = Integer.max((int) (admis * m * m / rank), PsplHMatrixPack.MINIMAL_SEP); 
    Dense d1 = new Dense(m, m, yStart, yStart + minimal_sep, func);
    Dense d2 = new Dense(d1.times(Matrix.random(m, rank)).getArray());
    QRDecomposition qr_ = d2.qr();
    return qr_.getQ();
  }

  public static Matrix getBasisVT (int xStart, int n, int rank, double admis, PsplHMatrixPack.DataFunction func) {
    int minimal_sep = Integer.max((int) (admis * n * n / rank), PsplHMatrixPack.MINIMAL_SEP);
    Dense d1 = new Dense(n, n, xStart + minimal_sep, xStart, func);
    Dense d2 = new Dense(Matrix.random(rank, n).times(d1).getArray());
    QRDecomposition qr_ = d2.transpose().qr();
    return qr_.getQ();
  }


}
