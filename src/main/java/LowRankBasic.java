
import java.io.*;
import java.nio.ByteBuffer;

import Jama.Matrix;
import Jama.QRDecomposition;
import Jama.SingularValueDecomposition;

public class LowRankBasic implements Block {
		
  private Matrix U;
  private Matrix VT;
  private int x_start = 0;
  private int y_start = 0;

  public LowRankBasic () {
    U = null; VT = null;
  }

  public LowRankBasic (int m, int n, int r) {
    U = new Matrix(m, r);
    VT = new Matrix(n, r);
  }

  public LowRankBasic (Matrix U, Matrix VT) {
    this.U = new Matrix(U.getArrayCopy());
    this.VT = new Matrix(VT.getArrayCopy());
  }

  public LowRankBasic (LowRank lr) {
    U = lr.getUS();
    VT = lr.getVT().toMatrix(U.getColumnDimension());
  }

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
  public int getRowDimension() 
  { return U == null ? 0 : U.getRowDimension(); }

  @Override
  public int getColumnDimension() 
  { return VT == null ? 0 : VT.getRowDimension(); }

  public int getRank()
  { return U == null ? 0 : U.getColumnDimension(); }
  
  public Matrix getU () 
  { return U; }

  public Matrix getVT () 
  { return VT; }

  @Override
  public Block_t getType() 
  { return Block_t.LOW_RANK; }

  @Override
  public Dense toDense() {
    if (U == null || VT == null)
    { return null; }
    else
    { return new Dense(U.times(VT.transpose()).getArray()); }
  }

  @Override
  public LowRank toLowRank() { 
    return new LowRank(U, Matrix.identity(U.getColumnDimension(), VT.getColumnDimension()), VT); 
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
    return this.toDense().minus(m).normF() / getColumnDimension() / getRowDimension();
  }

  @Override
  public double getCompressionRatio () {
    return (double) getRank() * (getColumnDimension() + getRowDimension()) / (getColumnDimension() * getRowDimension());
  }

  @Override
  public double getCompressionRatio_NoBasis () {
    return (double) getRank() * getRank() / (getColumnDimension() * getRowDimension());
  }

  @Override
  public String structure ()
  { return "LR " + Integer.toString(getRowDimension()) + " " + Integer.toString(getColumnDimension()) + " " + Integer.toString(getRank()) + "\n"; }

  @Override
  public void loadBinary (InputStream stream) throws IOException {
    int m = getRowDimension(), n = getColumnDimension(), r = getRank();
    byte data[];
    double data_ptr[][] = U.getArray();

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < r; j++) {
        data = stream.readNBytes(8);
        data_ptr[i][j] = ByteBuffer.wrap(data).getDouble(0); 
      }
    }

    data_ptr = VT.getArray();

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < r; j++) {
        data = stream.readNBytes(8);
        data_ptr[i][j] = ByteBuffer.wrap(data).getDouble(0); 
      }
    }
  }

  @Override
  public void writeBinary (OutputStream stream) throws IOException {
    int m = getRowDimension(), n = getColumnDimension(), r = getRank();
    byte data[] = new byte[8];
    double data_ptr[][] = U.getArray();

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < r; j++) {
        ByteBuffer.wrap(data).putDouble(0, data_ptr[i][j]);
        stream.write(data);
      }
    }

    data_ptr = VT.getArray();

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < r; j++) {
        ByteBuffer.wrap(data).putDouble(0, data_ptr[i][j]);
        stream.write(data);
      }
    }

  }

  @Override
  public void print (int w, int d)
  { U.print(w, d); VT.print(w, d); }

  @Override
  public Block LU () {
    System.out.println("error LU on LR");
    System.exit(-1);
    return null;
  }

  @Override
  public Block triangularSolve (Block b, boolean up_low) {
    return null;
  }


  @Override
  public Block GEMatrixMult (Block a, Block b, double alpha, double beta) {

    return null;
  }

  @Override
  public Block GEMatrixMult (Block a, Block b, double alpha, double beta, ClusterBasisProduct X, ClusterBasisProduct Y, ClusterBasisProduct Z, H2Approx Sa, H2Approx Sb, H2Approx Sc) {

    return null;
  }



  @Override
  public Block plusEquals (Block b) {
    return plusEquals(b.toLowRankBasic());
  }

  public LowRankBasic plusEquals (LowRankBasic lr) {

    if (U == null && VT == null)
    { U = new Matrix (lr.U.getArrayCopy()); VT = new Matrix (lr.VT.getArrayCopy()); return this; }

    int length = U.getColumnDimension() + lr.U.getColumnDimension();

    Matrix U_p = new Matrix (getRowDimension(), length);
    U_p.setMatrix(0, getRowDimension() - 1, 0, U.getColumnDimension() - 1, U);
    U_p.setMatrix(0, getRowDimension() - 1, U.getColumnDimension(), length - 1, lr.U);

    Matrix VT_p = new Matrix (getColumnDimension(), length);
    VT_p.setMatrix(0, getColumnDimension() - 1, 0, VT.getColumnDimension() - 1, VT);
    VT_p.setMatrix(0, getColumnDimension() - 1, VT.getColumnDimension(), length - 1, lr.VT);

    QRDecomposition qr_ = U_p.qr();

    Matrix Q = qr_.getQ().getMatrix(0, getRowDimension() - 1, 0, length - 1);
    Matrix R = qr_.getR().getMatrix(0, length - 1, 0, length - 1);

    SingularValueDecomposition svd_ = R.svd();
    double[] s = svd_.getSingularValues();
    int rank = 0;
    while(rank < length && s[rank] >= PsplHMatrixPack.EPI) 
    { rank++; }

    Matrix U_s = svd_.getU().getMatrix(0, length - 1, 0, rank - 1);
    Matrix S_s = svd_.getS().getMatrix(0, rank - 1, 0, rank - 1);
    Matrix V_s = svd_.getV().getMatrix(0, length - 1, 0, rank - 1);

    U = Q.times(U_s).times(S_s);
    VT = VT_p.times(V_s);
    return this;
  }

  @Override
  public Block scalarEquals (double s) {
    if (s != 1. && U != null)
    { U.timesEquals(s); }
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

  public LowRankBasic multLeft (Matrix m) {
    Matrix U_p = m.times(U);
    U = U_p;
    return this;
  }

  public LowRankBasic multRight (Matrix m) {
    Matrix VT_p = m.transpose().times(VT);
    VT = VT_p;
    return this;
  }



}
