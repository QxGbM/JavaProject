
import java.io.*;
import java.nio.ByteBuffer;

import Jama.LUDecomposition;
import Jama.Matrix;
import Jama.QRDecomposition;
import Jama.SingularValueDecomposition;

public class Dense extends Matrix implements Block 
{
  private static final long serialVersionUID = 1;
  private int x_start = 0;
  private int y_start = 0;

  public Dense (double[][] A)
  { super(A); }

  public Dense (double[][] A, int m, int n) 
  { super(A, m, n); }

  public Dense (double[] vals, int m) 
  { super(vals, m); }

  public Dense (int m, int n)
  { super(m, n); }

  public Dense (int m, int n, int y_start, int x_start, PsplHMatrixPack.dataFunction func)
  {
    super(m, n);
    this.x_start = x_start;
    this.y_start = y_start;
    double data[][] = getArray();

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++)
      { data[i][j] = func.body(y_start + i, x_start + j); }
    }

  }

  public Dense (int m, int n, double s)
  { super(m, n, s); }

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
  { return super.getRowDimension(); }

  @Override
  public int getColumnDimension() 
  { return super.getColumnDimension(); }

  @Override
  public Block_t getType()
  { return Block_t.DENSE; }

  @Override
  public Dense toDense()
  { return this; }

  @Override
  public LowRank toLowRank() {
    int m = getRowDimension(), n = getColumnDimension();
    int step = n > PsplHMatrixPack.rank * 4 ? PsplHMatrixPack.rank : (n < 4 ? 1 : n / 4), r = 0;

    boolean approx;
    Matrix Q, Y;
    QRDecomposition qr_;
    
    do {
      r += step; r = r >= n ? n : r;

      Matrix X = times(random(n, r));
      qr_ = X.qr();
      Q = qr_.getQ();
      Y = transpose().times(Q);
      Matrix A = Q.times(Y.transpose()).minus(this);
      double norm = A.normF() / (m * n);
      approx = norm <= PsplHMatrixPack.epi;

    } while (r < n && !approx);

    qr_ = Y.qr();
    Matrix V = qr_.getQ(), R = qr_.getR();
    SingularValueDecomposition svd_ = R.svd();

    ClusterBasis row_b = new ClusterBasis(Q.times(svd_.getV()), true);
    ClusterBasis col_b = new ClusterBasis(V.times(svd_.getU()), false);

    LowRank lr = new LowRank (row_b, svd_.getS(), col_b);
    lr.setClusterStart(x_start, y_start);

    return lr;
  }

  @Override
  public Hierarchical castHierarchical() {
    return null;
  }

  @Override
  public H2Matrix castH2Matrix() {
    return null;
  }

  public LowRank toLowRank_fromBasis (ClusterBasis row_basis, ClusterBasis col_basis_t) {
    Matrix S = row_basis.toMatrix().transpose().times(this).times(col_basis_t.toMatrix());
    return new LowRank(row_basis, S, col_basis_t);
  }

  public Matrix[] rsvd (int rank) {
    Matrix[] usv = new Matrix[3];

    Matrix X = times(random(getColumnDimension(), rank));
    QRDecomposition qr_ = X.qr();
    Matrix Q = qr_.getQ();
    Matrix Y = transpose().times(Q);

    qr_ = Y.qr();
    Matrix V = qr_.getQ(), R = qr_.getR();
    SingularValueDecomposition svd_ = R.svd();

    usv[0] = Q.times(svd_.getV());
    usv[1] = svd_.getS();
    usv[2] = V.times(svd_.getU());

    return usv;
  }

  public double rsvd_norm (Matrix[] usv) {
    return usv[0].times(usv[1]).times(usv[2].transpose()).minus(this).normF() / (double) (getRowDimension() * getColumnDimension());
  }

  public Dense transpose () {
    return new Dense(super.transpose().getArray());
  }

  @Override
  public boolean equals (Block b) {
    double norm = this.minus(b.toDense()).normF() / getColumnDimension() / getRowDimension();
    return norm <= PsplHMatrixPack.epi; 
  }

  @Override
  public double getCompressionRatio () 
  { return 1.; }

  @Override
  public double getCompressionRatio_NoBasis () 
  { return 1.; }

  @Override
  public String structure ()
  { return "D " + Integer.toString(getRowDimension()) + " " + Integer.toString(getColumnDimension()) + "\n"; }

  @Override
  public void loadBinary (InputStream stream) throws IOException
  {
    int m = getRowDimension(), n = getColumnDimension();
    byte data[];
    double data_ptr[][] = getArray();

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        data = stream.readNBytes(8 * m * n);
        data_ptr[i][j] = ByteBuffer.wrap(data).getDouble(0); 
      }
    }
  }

  @Override
  public void writeBinary (OutputStream stream) throws IOException {
    int m = getRowDimension(), n = getColumnDimension();
    byte data[] = new byte[8];
    double data_ptr[][] = getArray();

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) { 
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
  { super.print(w, d); }

  public static Dense readFromFile (String name) throws IOException {
    BufferedReader reader = new BufferedReader(new FileReader("bin/" + name + ".struct"));
    String str = reader.readLine();
    reader.close();

    if (str.startsWith("H")) {
      Hierarchical h = Hierarchical.readFromFile(name);
      return h.toDense();
    }
    else if (str.startsWith("LR")) {
      LowRank lr = LowRank.readFromFile(name);
      return lr.toDense();
    }
    else if (str.startsWith("D")) {
      String[] args = str.split("\\s+");
      int m = Integer.parseInt(args[1]), n = Integer.parseInt(args[2]);
      Dense d = new Dense(m, n);

      BufferedInputStream stream = new BufferedInputStream(new FileInputStream("bin/" + name + ".bin"));
      d.loadBinary(stream);
      stream.close();

      return d;
    }
    else
    { return null; }


  }

  @Override
  public void LU () {
    LUDecomposition lu_ = lu();
    Matrix L = lu_.getL(), U = lu_.getU();
    for (int i = 0; i < getRowDimension(); i++) {
      for (int j = 0; j < getColumnDimension(); j++) {
        set(i, j, i > j ? L.get(i, j) : U.get(i, j));
      }
    }
  }

  public Matrix getL() {
    Matrix L = new Matrix(getArray());
    for (int i = 0; i < getRowDimension(); i++) {
      L.set(i, i, 1);
      for (int j = i + 1; j < getColumnDimension(); j++) {
        L.set(i, j, 0);
      }
    }
    return L;
  }

  public Matrix getU() {
    Matrix U = new Matrix(getArray());
    for (int i = 0; i < getRowDimension(); i++) {
      for (int j = 0; j < i; j++) {
        U.set(i, j, 0);
      }
    }
    return U;
  }

  @Override
  public void triangularSolve (Block b, boolean up_low) {
    Matrix m;
    if (up_low)
    { m = solve(b.toDense().getL()); }
    else
    { m = solveTranspose(b.toDense().getU()); }
    setMatrix(0, getRowDimension() - 1, 0, getColumnDimension() - 1, m);
  }

  @Override
  public void GEMatrixMult (Block a, Block b, double alpha, double beta) {
    Matrix result = a.toDense().times(alpha).times(b.toDense());
    times(beta).plusEquals(result);
  }

  public Dense plusEquals (Dense d) {
    super.plusEquals(d);
    return this;
  }

  @Override
  public Block plusEquals (Block b) {
    return plusEquals(b.toDense());
  }

  public static Matrix getBasisU (int y_start, int m, int rank, double admis, PsplHMatrixPack.dataFunction func) {
    int minimal_sep = (int) (4 * admis * m); 
    //Dense d = new Dense(m, rank, y_start, y_start + minimal_sep, func);
    //QRDecomposition qr_ = d.qr();
    Dense d1 = new Dense(m, m, y_start, y_start + minimal_sep, func);
    Dense d2 = new Dense(d1.times(Matrix.random(m, rank)).getArray());
    QRDecomposition qr_ = d2.qr();
    return qr_.getQ();
  }

  public static Matrix getBasisVT (int x_start, int n, int rank, double admis, PsplHMatrixPack.dataFunction func) {
    int minimal_sep = (int) (4 * admis * n); 
    //Dense d = new Dense(rank, n, x_start + minimal_sep, x_start, func);
    //QRDecomposition qr_ = d.transpose().qr();
    Dense d1 = new Dense(n, n, x_start + minimal_sep, x_start, func);
    Dense d2 = new Dense(Matrix.random(rank, n).times(d1).getArray());
    QRDecomposition qr_ = d2.transpose().qr();
    return qr_.getQ();
  }


}
