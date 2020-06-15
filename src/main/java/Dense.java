
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
  public Dense toDense() {
    /*if (accm != null)
    { accum(accm); }*/
    return this; 
  }

  @Override
  public LowRank toLowRank() {
    int m = getRowDimension();
    int n = getColumnDimension();
    int step = n < 4 ? 1 : n / 4;
    int r = 0;
    step = n > 64 ? 16 : step;

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

  public LowRank toLowRankFromBasis (ClusterBasis rowBasis, ClusterBasis colBasisT) {
    Matrix s = rowBasis.toMatrix().transpose().times(this).times(colBasisT.toMatrix());
    return new LowRank(rowBasis, s, colBasisT);
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
    double[][] dataPtr = getArray();

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) { 
        ByteBuffer.wrap(data).putDouble(0, dataPtr[i][j]);
        stream.write(data);
      }
    }

  }

  @Override
  public Block getrf () {
    if (getAccumulator() != null)
    { accum(accm); }
    LUDecomposition lud = lu();
    Matrix l = lud.getL();
    Matrix u = lud.getU();
    for (int i = 0; i < getRowDimension(); i++) {
      for (int j = 0; j < getColumnDimension(); j++) {
        set(i, j, i > j ? l.get(i, j) : u.get(i, j));
      }
    }
    return this;
  }

  public Matrix getL() {
    Matrix l = new Matrix(getArrayCopy());
    for (int i = 0; i < getRowDimension(); i++) {
      l.set(i, i, 1);
      for (int j = i + 1; j < getColumnDimension(); j++) 
      { l.set(i, j, 0); }
    }
    return l;
  }

  public Matrix getU() {
    Matrix u = new Matrix(getArrayCopy());
    for (int i = 0; i < getRowDimension(); i++) {
      for (int j = 0; j < i; j++) 
      { u.set(i, j, 0); }
    }
    return u;
  }

  @Override
  public Block trsm (Block b, boolean lower) {
    if (getAccumulator() != null)
    { accum(accm); }
    if (b.castH2Matrix() != null)
    { return trsm(b.castH2Matrix(), lower); }
    else
    { return trsm(b.toDense(), lower); }
  }

  public Dense trsm (Dense d, boolean lower) {
    Matrix m = lower ? trsml(d, this) : trsmr(d, this);
    setMatrix(0, getRowDimension() - 1, 0, getColumnDimension() - 1, m);
    return this;
  }

  public Dense trsm (H2Matrix h, boolean lower) {
    Matrix m = lower ? trsml(h, this) : trsmr(h, this.transpose());
    setMatrix(0, getRowDimension() - 1, 0, getColumnDimension() - 1, m);
    return this;
  }

  public static Matrix trsml (Dense d, Matrix vec) {
    return d.getL().solve(vec);
  }

  public static Matrix trsmr (Dense d, Matrix vec) {
    return d.getU().solveTranspose(vec).transpose();
  }

  public static Matrix trsml (H2Matrix h, Matrix vec) {
    int m = h.getNRowBlocks();
    int n = h.getNColumnBlocks();
    Matrix[] vecP = h.getRowBasis().partitionMatrix(vec);
    for (int i = 0; i < n; i++) {
      Block b = h.getElement(i, i);
      if (b.castH2Matrix() != null)
      { vecP[i] = Dense.trsml(b.castH2Matrix(), vecP[i]); }
      else
      { vecP[i] = Dense.trsml(b.toDense(), vecP[i]); }
      for (int j = i + 1; j < m; j++) { 
        Block bJ = h.getElement(j, i);
        Matrix s;
        if (bJ.castH2Matrix() != null)
        { s = bJ.castH2Matrix().times(vecP[i], false); }
        else if (bJ.getType() == Block_t.LOW_RANK)
        { s = bJ.toLowRank().times(vecP[i], false); }
        else
        { s = bJ.toDense().times(vecP[i]); }
        vecP[j].minusEquals(s);
      }
    }

    Matrix ret = new Matrix(vec.getRowDimension(), vec.getColumnDimension());
    int y = 0;
    for (int i = 0; i < vecP.length; i++) {
      int yEnd = y + vecP[i].getRowDimension() - 1;
      ret.setMatrix(y, yEnd, 0, vec.getColumnDimension() - 1, vecP[i]);
      y = yEnd + 1;
    }
    return ret;
  }

  public static Matrix trsmr (H2Matrix h, Matrix vecT) {
    int m = h.getNRowBlocks();
    int n = h.getNColumnBlocks();
    Matrix[] vecP = h.getRowBasis().partitionMatrix(vecT);
    for (int i = 0; i < m; i++) {
      Block b = h.getElement(i, i);
      if (b.castH2Matrix() != null)
      { vecP[i] = Dense.trsmr(b.castH2Matrix(), vecP[i]); }
      else
      { vecP[i] = Dense.trsmr(b.toDense(), vecP[i].transpose()).transpose(); }
      for (int j = i + 1; j < n; j++) { 
        Block bJ = h.getElement(i, j);
        Matrix s;
        if (bJ.castH2Matrix() != null)
        { s = bJ.castH2Matrix().times(vecP[i], true); }
        else if (bJ.getType() == Block_t.LOW_RANK)
        { s = bJ.toLowRank().times(vecP[i], true); }
        else
        { s = bJ.toDense().transpose().times(vecP[i]); }
        vecP[j].minusEquals(s);
      }
    }

    Matrix ret = new Matrix(vecT.getRowDimension(), vecT.getColumnDimension());
    int y = 0;
    for (int i = 0; i < vecP.length; i++) {
      int yEnd = y + vecP[i].getRowDimension() - 1;
      ret.setMatrix(y, yEnd, 0, vecT.getColumnDimension() - 1, vecP[i]);
      y = yEnd + 1;
    }
    return ret;
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
    if (accm != null)
    { accm.scalarEquals(s); }
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
    Matrix r = super.times(d);
    return new Dense (r.getArrayCopy());
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
    int minimalSep = Integer.max((int) (admis * m * m / rank), PsplHMatrixPack.MINIMAL_SEP); 
    Dense d1 = new Dense(m, m, yStart, yStart + minimalSep, func);
    Dense d2 = new Dense(d1.times(Matrix.random(m, rank)).getArray());
    QRDecomposition qrd = d2.qr();
    return qrd.getQ();
  }

  public static Matrix getBasisVT (int xStart, int n, int rank, double admis, PsplHMatrixPack.DataFunction func) {
    int minimalSep = Integer.max((int) (admis * n * n / rank), PsplHMatrixPack.MINIMAL_SEP);
    Dense d1 = new Dense(n, n, xStart + minimalSep, xStart, func);
    Dense d2 = new Dense(Matrix.random(rank, n).times(d1).getArray());
    QRDecomposition qrd = d2.transpose().qr();
    return qrd.getQ();
  }


}
