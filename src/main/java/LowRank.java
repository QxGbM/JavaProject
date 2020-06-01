
import java.io.*;
import java.nio.ByteBuffer;

import Jama.Matrix;

public class LowRank implements Block {
		
  private Matrix S;
  private ClusterBasis U, VT;
  private int x_start = 0;
  private int y_start = 0;
  private LowRankBasic accm = null;

  public LowRank (int m, int n, int r) {
    U = new ClusterBasis(m, r);
    S = new Matrix(r, r);
    VT = new ClusterBasis(n, r);
  }

  public LowRank (ClusterBasis U, Matrix S, ClusterBasis VT) {
    this.U = U; this.VT = VT;

    if (U.getDimension() == S.getRowDimension() && VT.getDimension() == S.getColumnDimension())
    { this.S = U.toMatrix().transpose().times(S).times(VT.toMatrix()); }
    else if (U.getRank() >= S.getRowDimension() && VT.getRank() >= S.getColumnDimension())
    { this.S = S; }
    else { 
      System.out.println("Invalid Low-Rank Construction.");
      System.out.println("Dims: (" + U.getDimension() + ", " + U.getRank() + ") (" + S.getRowDimension() + ", " + S.getColumnDimension() + ") (" + VT.getDimension() + ", " + VT.getRank() + ").");
      System.exit(-1); 
    }
  }

  public LowRank (Matrix U, Matrix S, Matrix VT) {
    ClusterBasis row_b = new ClusterBasis(U);
    ClusterBasis col_b = new ClusterBasis(VT);
    this.U = row_b; this.VT = col_b;

    if (U.getRowDimension() == S.getRowDimension() && VT.getRowDimension() == S.getColumnDimension())
    { this.S = U.transpose().times(S).times(VT); }
    else if (U.getColumnDimension() == S.getRowDimension() && VT.getColumnDimension() == S.getColumnDimension())
    { this.S = S; }
    else
    { System.out.print("Invalid Low-Rank Construction."); System.exit(-1); }
  }

  public LowRank (ClusterBasis U, ClusterBasis VT, LowRank lr) {
    this.U = U; this.VT = VT;
    S = new ClusterBasisProduct(U, lr.U).getProduct().times(lr.S).times(new ClusterBasisProduct(VT, lr.VT).getProduct());
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
  { return U.getDimension(); }

  @Override
  public int getColumnDimension() 
  { return VT.getDimension(); }

  public int getRank()
  { return Integer.max(S.getRowDimension(), S.getColumnDimension()); }

  @Override
  public Block_t getType() 
  { return Block_t.LOW_RANK; }

  @Override
  public Dense toDense() {
    Matrix m1 = U.toMatrix(S.getRowDimension()).times(S);
    Matrix m2 = m1.times(VT.toMatrix(S.getColumnDimension()).transpose());
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
  public boolean equals (Block b) {
    double norm = compare(b.toDense());
    return norm <= PsplHMatrixPack.EPI; 
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
    double data_ptr[][] = U.toMatrix().getArray();

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < r; j++) {
        data = stream.readNBytes(8);
        data_ptr[i][j] = ByteBuffer.wrap(data).getDouble(0); 
      }
    }

    S = Matrix.identity(r, r);

    data_ptr = VT.toMatrix().getArray();

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
    double data_ptr[][] = U.toMatrix().getArray();

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < r; j++) {
        ByteBuffer.wrap(data).putDouble(0, data_ptr[i][j] * S.get(j, j));
        stream.write(data);
      }
    }

    data_ptr = VT.toMatrix().getArray();

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < r; j++) {
        ByteBuffer.wrap(data).putDouble(0, data_ptr[i][j]);
        stream.write(data);
      }
    }

  }

  @Override
  public void writeToFile (String name) throws IOException {
    File directory = new File("bin");
    if (!directory.exists())
    { directory.mkdir(); }
    
    BufferedWriter writer = new BufferedWriter(new FileWriter("bin/" + name + ".struct"));
    String struct = structure();
    writer.write(struct);
    writer.flush();
    writer.close();

    BufferedOutputStream stream = new BufferedOutputStream(new FileOutputStream("bin/" + name + ".bin"));
    writeBinary(stream);
    stream.flush();
    stream.close();
  }

  @Override
  public void print (int w, int d)
  { U.toMatrix().print(w, d); S.print(w, d); VT.toMatrix().print(w, d); }

  @Override
  public Block LU () {
    System.out.println("error LU on LR");
    System.exit(-1);
    return null;
  }

  @Override
  public Block triangularSolve (Block b, boolean up_low) {
    if (getAccumulator() != null)
    { accum(accm); }
    return triangularSolve(b.toDense(), up_low); // TODO
  }

  public LowRank triangularSolve (Dense d, boolean up_low) {

    if (up_low) {
      Matrix vt = getVT().toMatrix(S.getColumnDimension()).times(S.transpose());
      Matrix vt_prime = d.getU().solveTranspose(vt.transpose());
      ClusterBasisProduct vt_proj = VT.updateAdditionalBasis(vt_prime);
      S = vt_proj.getProduct().transpose();
    }
    else {
      Matrix u = getU().toMatrix(S.getRowDimension()).times(S);
      Matrix u_prime = d.getL().solve(u);
      ClusterBasisProduct u_proj = U.updateAdditionalBasis(u_prime);
      S = u_proj.getProduct();
    }

    return this;
  }

  @Override
  public Block GEMatrixMult (Block a, Block b, double alpha, double beta) {
    if (a.castH2Matrix() != null || b.castH2Matrix() != null) { 
      /*H2Matrix h = new H2Matrix(this); h.GEMatrixMult(a, b, alpha, beta);
      LowRank lr = h.toLowRank(); S = lr.S;*/
      scalarEquals(beta);
      Block c = a.times(b);
      c.scalarEquals(alpha);
      plusEquals(c);
    }
    else if (a.getType() == Block_t.DENSE) {
      scalarEquals(beta);
      Block c = a.toDense().times(b);
      c.scalarEquals(alpha);
      plusEquals(c);
    }
    else if (a.getType() == Block_t.LOW_RANK) {
      scalarEquals(beta);
      Block c = a.toLowRank().times(b);
      c.scalarEquals(alpha);
      plusEquals(c);
    }

    return this;
  }

  @Override
  public Block GEMatrixMult (Block a, Block b, double alpha, double beta, ClusterBasisProduct X, ClusterBasisProduct Y, ClusterBasisProduct Z, H2Approx Sa, H2Approx Sb, H2Approx Sc) {
    if (a.getType() == Block_t.LOW_RANK) 
    { GEMatrixMult(a.toLowRank(), b, alpha, beta, X, Y, Z, Sa, Sb, Sc); }
    else if (b.getType() == Block_t.LOW_RANK)
    { GEMatrixMult(a, b.toLowRank(), alpha, beta, X, Y, Z, Sa, Sb, Sc); }
    else if (a.getType() == Block_t.DENSE && b.getType() == Block_t.DENSE)
    { GEMatrixMult(a.toDense(), b.toDense(), alpha, beta, X, Y, Z, Sa, Sb, Sc); }
    else 
    { GEMatrixMult(a.castH2Matrix(), b.castH2Matrix(), alpha, beta, X, Y, Z, Sa, Sb, Sc); }

    return this;
  }

  public LowRank GEMatrixMult (LowRank a, Block b, double alpha, double beta, ClusterBasisProduct X, ClusterBasisProduct Y, ClusterBasisProduct Z, H2Approx Sa, H2Approx Sb, H2Approx Sc) {
    scalarEquals(beta);
    Matrix m = X.getProduct().times(a.getS()).times(Sb.getS()).times(alpha);
    S.plusEquals(m);
    return this;
  }

  public LowRank GEMatrixMult (Block a, LowRank b, double alpha, double beta, ClusterBasisProduct X, ClusterBasisProduct Y, ClusterBasisProduct Z, H2Approx Sa, H2Approx Sb, H2Approx Sc) {
    scalarEquals(beta);
    Matrix m = Sa.getS().times(b.getS()).times(Z.getProduct()).times(alpha);
    this.S.plusEquals(m);
    return this;
  }

  public LowRank GEMatrixMult (Dense a, Dense b, double alpha, double beta, ClusterBasisProduct X, ClusterBasisProduct Y, ClusterBasisProduct Z, H2Approx Sa, H2Approx Sb, H2Approx Sc) {
    scalarEquals(beta);
    Matrix m = U.toMatrix().transpose().times(a).times(b).times(VT.toMatrix()).times(alpha);
    this.S.plusEquals(m);
    return this;
  }

  public LowRank GEMatrixMult (H2Matrix a, H2Matrix b, double alpha, double beta, ClusterBasisProduct X, ClusterBasisProduct Y, ClusterBasisProduct Z, H2Approx Sa, H2Approx Sb, H2Approx Sc) {
    scalarEquals(beta);
    H2Matrix temp = new H2Matrix(this);
    temp.GEMatrixMult(a, b, alpha, 1., X, Y, Z, Sa, Sb, Sc);
    LowRank lr = temp.toLowRank();
    S = lr.S;
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

  public LowRank plusEquals (ClusterBasisProduct X, ClusterBasisProduct Y, Matrix S_prime) {
    Matrix a = X == null ? S_prime : X.getProduct().times(S_prime);
    Matrix b = Y == null ? a : a.times(Y.getProduct());
    S.plusEquals(b);
    return this;
  }

  @Override
  public Block scalarEquals (double s) {
    if (s != 1.)
    { S.timesEquals(s); }
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
    { System.out.println("Error partition."); System.exit(-1); return null; }
  }

  public LowRankBasic times (Dense d) {
    int rank = S.getColumnDimension();
    Matrix u = getUS();
    Matrix vt = d.transpose().times(VT.toMatrix(rank));
    return new LowRankBasic (u, vt);
  }

  public LowRankBasic times (LowRank lr) {
    Matrix p = new ClusterBasisProduct(VT, lr.U).getProduct();
    int left = S.getColumnDimension();
    int right = lr.S.getRowDimension();
    Matrix q = p.getMatrix(0, left - 1, 0, right - 1);
    Matrix u = getUS().times(q).times(lr.S);
    Matrix vt = lr.VT.toMatrix(lr.S.getColumnDimension());
    return new LowRankBasic (u, vt);
  }

  public LowRankBasic times (H2Matrix h) {
    Matrix us = getUS();
    Matrix vt = VT.h2matrixTimes(h, true);
    return new LowRankBasic(us, vt);
  }

  @Override
  public Block accum (LowRankBasic accm) {
    ClusterBasisProduct u_proj = U.updateAdditionalBasis(accm.getU());
    ClusterBasisProduct vt_proj = VT.updateAdditionalBasis(accm.getVT());

    Matrix S_prime = new Matrix(U.getRank(), VT.getRank());
    S_prime.setMatrix(0, S.getRowDimension() - 1, 0, S.getColumnDimension() - 1, S);

    Matrix U_proj = u_proj.getProduct();
    Matrix V_proj = vt_proj.getProduct().transpose();
    S_prime.plusEquals(U_proj.times(V_proj));

    S = S_prime;
    return this;
  }


  public ClusterBasis getU ()
  { return U; }

  public Matrix getS ()
  { return S; }
  
  public ClusterBasis getVT ()
  { return VT; }

  public Matrix getUS ()
  { return U.toMatrix(S.getRowDimension()).times(S); }


  public static LowRank readFromFile (String name) throws IOException {
    BufferedReader reader = new BufferedReader(new FileReader("bin/" + name + ".struct"));
    String str = reader.readLine();
    reader.close();

    if (str.startsWith("H")) {
      Hierarchical h = Hierarchical.readFromFile(name);
      return h.toLowRank();
    }
    else if (str.startsWith("D")) {
      Dense d = Dense.readFromFile(name);
      return d.toLowRank();
    }
    else if (str.startsWith("LR")) {
      String[] args = str.split("\\s+");
      int m = Integer.parseInt(args[1]), n = Integer.parseInt(args[2]), r = Integer.parseInt(args[3]);
      LowRank lr = new LowRank(m, n, r);

      BufferedInputStream stream = new BufferedInputStream(new FileInputStream("bin/" + name + ".bin"));
      lr.loadBinary(stream);
      stream.close();

      return lr;
    }
    else
    { return null; }
    
  }

}
