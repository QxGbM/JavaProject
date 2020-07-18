
import java.io.*;
import java.nio.ByteBuffer;

import Jama.Matrix;
import Jama.QRDecomposition;
import Jama.SingularValueDecomposition;

public class LowRankBasic implements Block {
		
  private Matrix u;
  private Matrix vt;

  public LowRankBasic () {
    u = null; vt = null;
  }

  public LowRankBasic (int m, int n, int r) {
    u = new Matrix(m, r);
    vt = new Matrix(n, r);
  }

  public LowRankBasic (Matrix u, Matrix vt) {
    this.u = new Matrix(u.getArrayCopy());
    this.vt = new Matrix(vt.getArrayCopy());
  }

  public LowRankBasic (LowRank lr) {
    u = lr.getUS();
    vt = lr.getVT().toMatrix(u.getColumnDimension());
  }


  @Override
  public int getRowDimension() 
  { return u == null ? 0 : u.getRowDimension(); }

  @Override
  public int getColumnDimension() 
  { return vt == null ? 0 : vt.getRowDimension(); }

  public int getRank()
  { return u == null ? 0 : u.getColumnDimension(); }
  
  public Matrix getU () 
  { return u; }

  public Matrix getVT () 
  { return vt; }

  @Override
  public Block_t getType() 
  { return Block_t.LOW_RANK; }

  @Override
  public Dense toDense() {
    if (u == null || vt == null)
    { return null; }
    else
    { return new Dense(u.times(vt.transpose()).getArray()); }
  }

  @Override
  public LowRank toLowRank() { 
    return new LowRank(u, Matrix.identity(u.getColumnDimension(), vt.getColumnDimension()), vt); 
  }
  
  @Override
  public LowRankBasic toLowRankBasic () { 
    return this;
  }

  @Override
  public Hierarchical castHierarchical() {
    return null;
  }

  @Override
  public H2Matrix castH2Matrix() {
    return null;
  }

  @Override
  public void setAccumulator (LowRankBasic accm) {
    // no accm needed
  }

  @Override
  public LowRankBasic getAccumulator() {
    return null;
  }

  @Override
  public double compare (Matrix m) {
    return this.toDense().minus(m).normF();
  }

  @Override
  public double getCompressionRatio () {
    return (double) getRank() * (getColumnDimension() + getRowDimension()) / (getColumnDimension() * getRowDimension());
  }

  @Override
  public double getCompressionRatioNoBasis () {
    return (double) getRank() * getRank() / (getColumnDimension() * getRowDimension());
  }

  @Override
  public String structure ()
  { return "LR " + Integer.toString(getRowDimension()) + " " + Integer.toString(getColumnDimension()) + " " + Integer.toString(getRank()) + "\n"; }

  @Override
  public void loadBinary (InputStream stream) throws IOException {
    int m = getRowDimension();
    int n = getColumnDimension();
    int r = getRank();
    byte[] data;
    double[][] dataPtr = u.getArray();

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < r; j++) {
        data = stream.readNBytes(8);
        dataPtr[i][j] = ByteBuffer.wrap(data).getDouble(0); 
      }
    }

    dataPtr = vt.getArray();

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < r; j++) {
        data = stream.readNBytes(8);
        dataPtr[i][j] = ByteBuffer.wrap(data).getDouble(0); 
      }
    }
  }

  @Override
  public void writeBinary (OutputStream stream) throws IOException {
    int m = getRowDimension();
    int n = getColumnDimension();
    int r = getRank();
    byte[] data = new byte[8];
    double[][] dataPtr = u.getArray();

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < r; j++) {
        ByteBuffer.wrap(data).putDouble(0, dataPtr[i][j]);
        stream.write(data);
      }
    }

    dataPtr = vt.getArray();

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < r; j++) {
        ByteBuffer.wrap(data).putDouble(0, dataPtr[i][j]);
        stream.write(data);
      }
    }

  }

  @Override
  public void print (int w, int d)
  { u.print(w, d); vt.print(w, d); }

  @Override
  public Block getrf () {
    PsplHMatrixPack.errorOut("error Lu on LR");
    return null;
  }

  @Override
  public Block trsm (Block b, boolean lower) {
    return null;
  }


  @Override
  public Block gemm (Block a, Block b, double alpha, double beta) {
    return null;
  }


  @Override
  public Block plusEquals (Block b) {
    return plusEquals(b.toLowRankBasic());
  }

  public LowRankBasic plusEquals (LowRankBasic lr) {

    if (u == null || vt == null)
    { u = new Matrix (lr.u.getArrayCopy()); vt = new Matrix (lr.vt.getArrayCopy()); return this; }

    int length = u.getColumnDimension() + lr.u.getColumnDimension();

    Matrix uPrime = new Matrix (getRowDimension(), length);
    uPrime.setMatrix(0, getRowDimension() - 1, 0, u.getColumnDimension() - 1, u);
    uPrime.setMatrix(0, getRowDimension() - 1, u.getColumnDimension(), length - 1, lr.u);

    Matrix vtPrime = new Matrix (getColumnDimension(), length);
    vtPrime.setMatrix(0, getColumnDimension() - 1, 0, vt.getColumnDimension() - 1, vt);
    vtPrime.setMatrix(0, getColumnDimension() - 1, vt.getColumnDimension(), length - 1, lr.vt);

    QRDecomposition qrd = uPrime.qr();

    Matrix q = qrd.getQ().getMatrix(0, getRowDimension() - 1, 0, length - 1);
    Matrix r = qrd.getR().getMatrix(0, length - 1, 0, length - 1);

    SingularValueDecomposition svdd = r.svd();
    double[] s = svdd.getSingularValues();
    int rank = 0;
    while(rank < length && s[rank] >= PsplHMatrixPack.EPI) 
    { rank++; }

    Matrix uS = svdd.getU().getMatrix(0, length - 1, 0, rank - 1);
    Matrix sS = svdd.getS().getMatrix(0, rank - 1, 0, rank - 1);
    Matrix vS = svdd.getV().getMatrix(0, length - 1, 0, rank - 1);

    u = q.times(uS).times(sS);
    vt = vtPrime.times(vS);
    return this;
  }

  @Override
  public Block scalarEquals (double s) {
    if (s != 1. && u != null)
    { u.timesEquals(s); }
    return this;
  }

  @Override
  public Block times (Block b) {
    return null;
  }

  @Override
  public Block accum (LowRankBasic accm) {
    return plusEquals(accm);
  }

  @Override
  public Block copyBlock () {
    return new LowRankBasic(u, vt);
  }

  public LowRankBasic multLeft (Matrix m) {
    Matrix uPrime = m.times(u);
    u = uPrime;
    return this;
  }

  public LowRankBasic multRight (Matrix m) {
    Matrix vtPrime = m.transpose().times(vt);
    vt = vtPrime;
    return this;
  }



}
