
import java.io.*;
import java.nio.ByteBuffer;

import Jama.Matrix;

public class LowRank implements Block {
		
  private Matrix s;
  private ClusterBasis u;
  private ClusterBasis vt;
  private LowRankBasic accm = null;

  public LowRank (int m, int n, int r) {
    u = new ClusterBasis(m, r);
    s = new Matrix(r, r);
    vt = new ClusterBasis(n, r);
  }

  public LowRank (ClusterBasis u, Matrix s, ClusterBasis vt) {
    this.u = u; this.vt = vt;

    if (u.getDimension() == s.getRowDimension() && vt.getDimension() == s.getColumnDimension())
    { this.s = u.toMatrix().transpose().times(s).times(vt.toMatrix()); }
    else if (u.getRank() >= s.getRowDimension() && vt.getRank() >= s.getColumnDimension())
    { this.s = s; }
    else 
    { PsplHMatrixPack.errorOut("Invalid Low-Rank Construction."); }
  }

  public LowRank (Matrix u, Matrix s, Matrix vt) {
    ClusterBasis rowB = new ClusterBasis(u);
    ClusterBasis colB = new ClusterBasis(vt);
    this.u = rowB; this.vt = colB;

    if (u.getRowDimension() == s.getRowDimension() && vt.getRowDimension() == s.getColumnDimension())
    { this.s = u.transpose().times(s).times(vt); }
    else if (u.getColumnDimension() == s.getRowDimension() && vt.getColumnDimension() == s.getColumnDimension())
    { this.s = s; }
    else
    { PsplHMatrixPack.errorOut("Invalid Low-Rank Construction."); }
  }

  public LowRank (ClusterBasis u, ClusterBasis vt, LowRank lr) {
    this.u = u; this.vt = vt;
    s = new ClusterBasisProduct(u, lr.u).getProduct().times(lr.s).times(new ClusterBasisProduct(vt, lr.vt).getProduct());
  }


  @Override
  public int getRowDimension() 
  { return u.getDimension(); }

  @Override
  public int getColumnDimension() 
  { return vt.getDimension(); }

  public int getRank()
  { return Integer.max(s.getRowDimension(), s.getColumnDimension()); }

  @Override
  public Block_t getType() 
  { return Block_t.LOW_RANK; }

  @Override
  public Dense toDense() {
    Matrix m1 = u.toMatrix(s.getRowDimension()).times(s);
    Matrix m2 = m1.times(vt.toMatrix(s.getColumnDimension()).transpose());
    return new Dense(m2.getArray());
  }

  @Override
  public LowRank toLowRank() 
  { return this; }
  
  @Override
  public LowRankBasic toLowRankBasic () { 
    return new LowRankBasic(this);
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
    double[][] dataPtr = u.toMatrix().getArray();

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < r; j++) {
        data = stream.readNBytes(8);
        dataPtr[i][j] = ByteBuffer.wrap(data).getDouble(0); 
      }
    }

    s = Matrix.identity(r, r);

    dataPtr = vt.toMatrix().getArray();

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
    double[][] dataPtr = u.toMatrix().getArray();

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < r; j++) {
        ByteBuffer.wrap(data).putDouble(0, dataPtr[i][j] * s.get(j, j));
        stream.write(data);
      }
    }

    dataPtr = vt.toMatrix().getArray();

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < r; j++) {
        ByteBuffer.wrap(data).putDouble(0, dataPtr[i][j]);
        stream.write(data);
      }
    }

  }

  @Override
  public void print (int w, int d)
  { u.toMatrix().print(w, d); s.print(w, d); vt.toMatrix().print(w, d); }

  @Override
  public Block getrf () {
    PsplHMatrixPack.errorOut("error Lu on LR");
    return null;
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

  public LowRank trsm (Dense d, boolean lower) {

    if (lower) {
      Matrix up = getU().toMatrix(s.getRowDimension()).times(s);
      Matrix uPrime = Dense.trsml(d, up);
      ClusterBasisProduct uProj = u.updateAdditionalBasis(uPrime);
      s = uProj.getProduct();
    }
    else {
      Matrix vtp = getVT().toMatrix(s.getColumnDimension()).times(s.transpose());
      Matrix vtPrime = Dense.trsmr(d, vtp.transpose());
      ClusterBasisProduct vtProj = vt.updateAdditionalBasis(vtPrime);
      s = vtProj.getProduct().transpose();
    }

    return this;
  }

  public LowRank trsm (H2Matrix h, boolean lower) {
    
    if (lower) {
      Matrix up = getU().toMatrix(s.getRowDimension()).times(s);
      Matrix uPrime = Dense.trsml(h, up);
      ClusterBasisProduct uProj = u.updateAdditionalBasis(uPrime);
      s = uProj.getProduct();
    }
    else {
      Matrix vtp = getVT().toMatrix(s.getColumnDimension()).times(s.transpose());
      Matrix vtPrime = Dense.trsmr(h, vtp);
      ClusterBasisProduct vtProj = vt.updateAdditionalBasis(vtPrime);
      s = vtProj.getProduct().transpose();
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
    LowRank lr = b.toLowRank();
    return plusEquals(lr);
  }

  public LowRank plusEquals (LowRank lr) {
    if (accm == null)
    { accm = new LowRankBasic(); }
    accm.plusEquals(lr.toLowRankBasic());
    return this;
  }

  @Override
  public Block scalarEquals (double a) {
    if (accm != null)
    { accm.scalarEquals(a); }
    if (a != 1.)
    { s.timesEquals(a); }
    return this;
  }

  @Override
  public Block times (Block b) {
    if (b.castH2Matrix() != null)
    { return times(b.castH2Matrix()); }
    else if (b.getType() == Block_t.LOW_RANK)
    { return times(b.toLowRank()); }
    else if (b.getType() == Block_t.DENSE)
    { return times(b.toDense()); }
    else
    { PsplHMatrixPack.errorOut("error partition"); return null; }
  }

  public LowRankBasic times (Dense d) {
    int rank = s.getColumnDimension();
    Matrix us = getUS();
    Matrix vtp = d.transpose().times(vt.toMatrix(rank));
    return new LowRankBasic (us, vtp);
  }

  public LowRankBasic times (LowRank lr) {
    Matrix p = new ClusterBasisProduct(vt, lr.u).getProduct();
    int left = s.getColumnDimension();
    int right = lr.s.getRowDimension();
    Matrix q = p.getMatrix(0, left - 1, 0, right - 1);
    Matrix uss = getUS().times(q).times(lr.s);
    Matrix vtp = lr.vt.toMatrix(lr.s.getColumnDimension());
    return new LowRankBasic (uss, vtp);
  }

  public LowRankBasic times (H2Matrix h) {
    Matrix us = getUS();
    Matrix vtp = vt.h2matrixTimes(h, true);
    return new LowRankBasic(us, vtp);
  }

  public Matrix times (Matrix vec, boolean transpose) {
    int rank = s.getColumnDimension();
    Matrix us = getUS();
    Matrix vtp = vec.transpose().times(vt.toMatrix(rank));
    return transpose ? vtp.times(us.transpose()) : us.times(vtp.transpose());
  }

  @Override
  public Block accum (LowRankBasic accm) {
    ClusterBasisProduct uProj = u.updateAdditionalBasis(accm.getU());
    ClusterBasisProduct vtProj = vt.updateAdditionalBasis(accm.getVT());

    Matrix sPrime = new Matrix(u.getRank(), vt.getRank());
    sPrime.setMatrix(0, s.getRowDimension() - 1, 0, s.getColumnDimension() - 1, s);

    Matrix uProjP = uProj.getProduct();
    Matrix vProj = vtProj.getProduct().transpose();
    sPrime.plusEquals(uProjP.times(vProj));

    s = sPrime;
    return this;
  }

  @Override
  public Block copyBlock () {
    LowRank lr = new LowRank(u, s.copy(), vt);
    if (accm != null)
    { lr.setAccumulator(accm.copyBlock().toLowRankBasic()); }
    return lr;
  }

  public ClusterBasis getU ()
  { return u; }

  public Matrix getS ()
  { return s; }
  
  public ClusterBasis getVT ()
  { return vt; }

  public Matrix getUS ()
  { return u.toMatrix(s.getRowDimension()).times(s); }


}
