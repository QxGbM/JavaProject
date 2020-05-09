
import java.io.*;
import java.nio.ByteBuffer;

import Jama.Matrix;

public class LowRank implements Block {

  enum LR_FORM { U_S_V, US_V, U_SV };
		
  private Matrix S;
  private ClusterBasis U, VT;
  private LR_FORM form;
  private int x_start = 0;
  private int y_start = 0;

  public LowRank (int m, int n, int r) {
    U = new ClusterBasis(m, r);
    S = new Matrix(r, r);
    VT = new ClusterBasis(n, r);
    form = LR_FORM.U_S_V;
  }

  public LowRank (ClusterBasis U, Matrix S, ClusterBasis VT) {
    this.U = U; this.VT = VT;

    if (U.getDimension() == S.getRowDimension() && VT.getDimension() == S.getColumnDimension())
    { this.S = U.toMatrix().transpose().times(S).times(VT.toMatrix()); }
    else if (U.getRank() == S.getRowDimension() && VT.getRank() == S.getColumnDimension())
    { this.S = S; }
    else
    { 
      System.out.println("Invalid Low-Rank Construction.");
      System.out.println("Dims: (" + U.getDimension() + ", " + U.getRank() + ") (" + S.getRowDimension() + ", " + S.getColumnDimension() + ") (" + VT.getDimension() + ", " + VT.getRank() + ").");
      System.exit(-1); 
    }
    form = LR_FORM.U_S_V;
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
    form = LR_FORM.U_S_V;
  }

  public LowRank (ClusterBasis U, ClusterBasis VT, LowRank lr) {
    this.U = U; this.VT = VT;
    if (lr.form == LR_FORM.US_V)
    { S = U.toMatrix().transpose().times(lr.S).times(new ClusterBasisProduct(VT, lr.VT).getProduct()); }
    else if (lr.form == LR_FORM.U_SV)
    { S = new ClusterBasisProduct(U, lr.U).getProduct().times(lr.S).times(VT.toMatrix()); }
    else
    { S = new ClusterBasisProduct(U, lr.U).getProduct().times(lr.S).times(new ClusterBasisProduct(VT, lr.VT).getProduct()); }
    form = LR_FORM.U_S_V;
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
  { return form == LR_FORM.US_V ? S.getRowDimension() : U.getDimension(); }

  @Override
  public int getColumnDimension() 
  { return form == LR_FORM.U_SV ? S.getColumnDimension() : VT.getDimension(); }

  public int getRank()
  { return form == LR_FORM.US_V ? S.getColumnDimension() : S.getRowDimension(); }

  @Override
  public Block_t getType() 
  { return Block_t.LOW_RANK; }

  @Override
  public Dense toDense() {
    Matrix m1 = form == LR_FORM.US_V ? S : U.toMatrix(S.getRowDimension()).times(S);
    Matrix m2 = form == LR_FORM.U_SV ? m1 : m1.times(VT.toMatrix(S.getColumnDimension()).transpose());
    return new Dense(m2.getArray());
  }

  @Override
  public LowRank toLowRank() 
  { return this; }

  @Override
  public Hierarchical castHierarchical() {
    return null;
  }

  @Override
  public H2Matrix castH2Matrix() {
    return null;
  }

  @Override
  public boolean equals (Block b) {
    double norm = compare(b.toDense());
    return norm <= PsplHMatrixPack.epi; 
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
    return triangularSolve(b.toDense(), up_low);
  }

  public LowRank triangularSolve (Dense d, boolean up_low) {

    if (up_low) {
      Matrix vt = getVT().toMatrix(S.getColumnDimension()).times(S.transpose());
      Matrix vt_prime = d.getU().solveTranspose(vt.transpose());
      Matrix vt_new = VT.updateAdditionalBasis(vt_prime);
      S = vt_prime.transpose().times(vt_new);
    }
    else {
      Matrix u = getU().toMatrix(S.getRowDimension()).times(S);
      Matrix u_prime = d.getL().solve(u);
      Matrix u_new = U.updateAdditionalBasis(u_prime);
      S = u_new.transpose().times(u_prime);
    }

    return this;
  }

  @Override
  public Block GEMatrixMult (Block a, Block b, double alpha, double beta) {
    if (a.castH2Matrix() != null || b.castH2Matrix() != null) { 
      H2Matrix h = new H2Matrix(this); h.GEMatrixMult(a, b, alpha, beta);
      LowRank lr = h.toLowRank(); S = lr.S;
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
    if (a.getType() == Block_t.LOW_RANK && form != LR_FORM.US_V) 
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
    if (form == LR_FORM.U_SV) {
      //H2Approx Sb_prime = new H2Approx(Sb, VT, false);
      //Matrix m = X.getProduct().times(a.getS()).times(Sb_prime.getS()).times(alpha);
      Matrix m = X.getProduct().times(a.getS()).times(a.getVT().toMatrix().transpose()).times(b.toDense()).times(alpha);
      this.S.plusEquals(m);
    }
    else {
      Matrix m = X.getProduct().times(a.getS()).times(Sb.getS()).times(alpha);
      this.S.plusEquals(m);
    }
    return this;
  }

  public LowRank GEMatrixMult (Block a, LowRank b, double alpha, double beta, ClusterBasisProduct X, ClusterBasisProduct Y, ClusterBasisProduct Z, H2Approx Sa, H2Approx Sb, H2Approx Sc) {
    scalarEquals(beta);
    if (form == LR_FORM.US_V) {
      //H2Approx Sa_prime = new H2Approx(Sa, U, true);
      //Matrix m = Sa_prime.getS().times(b.getS()).times(Z.getProduct()).times(alpha);
      Matrix m = a.toDense().times(b.getU().toMatrix()).times(b.getS()).times(Z.getProduct()).times(alpha);
      this.S.plusEquals(m);
    }
    else {
      Matrix m = Sa.getS().times(b.getS()).times(Z.getProduct()).times(alpha);
      this.S.plusEquals(m);
    }
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
  public void unshareBasis (boolean row_col) {
    if (row_col && form == LR_FORM.U_S_V) 
    { S = U.toMatrix().times(S); form = LR_FORM.US_V; }
    else if (!row_col && form == LR_FORM.U_S_V)
    { S = S.times(VT.toMatrix().transpose()); form = LR_FORM.U_SV; }
  }

  public LowRank plusEquals (LowRank lr) {
    /*boolean U_equal = lr.U.compare(U), VT_equal = lr.VT.compare(VT);
    ClusterBasisProduct X = U_equal ? null : new ClusterBasisProduct(U, lr.U);
    ClusterBasisProduct Y = VT_equal ? null : new ClusterBasisProduct(lr.VT, VT);
    return plusEquals(X, Y, lr.S);*/

    Matrix row_f = lr.getU().toMatrix().times(lr.getS());
    Matrix col_f = lr.getVT().toMatrix().times(lr.getS().transpose());

    Matrix u_new = U.updateAdditionalBasis(row_f);
    Matrix vt_new = VT.updateAdditionalBasis(col_f);

    Matrix S_prime = new Matrix(U.getRank(), VT.getRank());
    S_prime.setMatrix(0, S.getRowDimension() - 1, 0, S.getColumnDimension() - 1, S);

    Matrix U_proj = u_new.transpose().times(row_f);
    Matrix V_proj = lr.VT.toMatrix().transpose().times(vt_new);
    S_prime.plusEquals(U_proj.times(V_proj));

    S = S_prime;
    return this;
  }

  public LowRank plusEquals (ClusterBasisProduct X, ClusterBasisProduct Y, Matrix S_prime) {
    Matrix a = form == LR_FORM.US_V ? U.toMatrix().times(S_prime) : (X == null ? S_prime : X.getProduct().times(S_prime));
    Matrix b = form == LR_FORM.U_SV ? a.times(VT.toMatrix().transpose()) : (Y == null ? a : a.times(Y.getProduct()));
    S.plusEquals(b);
    return this;
  }

  @Override
  public Block plusEquals (Block b) {
    LowRank lr = b.toLowRank();
    return plusEquals(lr);
  }

  @Override
  public Block scalarEquals (double s) {
    if (s != 1.)
    { S.timesEquals(s); }
    return this;
  }

  public LowRank times (Dense d) {
    Matrix s_prime = new Matrix(getS().getArrayCopy());
    ClusterBasis rb = new ClusterBasis(getU().toMatrix(s_prime.getRowDimension()));
    ClusterBasis cb = new ClusterBasis(d.transpose().times(getVT().toMatrix(s_prime.getColumnDimension())));
    return new LowRank (rb, s_prime, cb);
  }

  public LowRank times (LowRank lr) {
    Matrix left_V = getS().times(getVT().toMatrix(getS().getColumnDimension()).transpose());
    Matrix right_U = lr.getU().toMatrix(lr.getS().getRowDimension()).times(lr.getS());
    Matrix s_prime = left_V.times(right_U);
    ClusterBasis rb = new ClusterBasis(getU().toMatrix(s_prime.getRowDimension()));
    ClusterBasis cb = new ClusterBasis(lr.getVT().toMatrix(s_prime.getColumnDimension()));
    return new LowRank (rb, s_prime, cb);
  }

  public Block times (Block b) {
    if (b.getType() == Block_t.LOW_RANK)
    { return times(b.toLowRank()); }
    else if (b.getType() == Block_t.DENSE)
    { return times(b.toDense()); }
    else
    { System.out.println("Error partition."); System.exit(-1); return null; }
  }


  public ClusterBasis getU ()
  { return U; }

  public Matrix getS ()
  { return form == LR_FORM.U_S_V ? S : (form == LR_FORM.US_V ? U.toMatrix().transpose().times(S) : S.times(VT.toMatrix())); }
  
  public ClusterBasis getVT ()
  { return VT; }

  public Matrix[] getPair () { 
    if (form == LR_FORM.US_V)
    { return new Matrix[] { S, VT.toMatrix().transpose() }; }
    else if (form == LR_FORM.U_SV)
    { return new Matrix[] { U.toMatrix(), S }; }
    else
    { return new Matrix[] { U.toMatrix().times(S), VT.toMatrix().transpose() };} 
  }

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
