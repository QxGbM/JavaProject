
import java.io.*;
import java.nio.ByteBuffer;

import Jama.Matrix;
import Jama.QRDecomposition;
import Jama.SingularValueDecomposition;

public class Dense extends Matrix implements Block 
{
  private static final long serialVersionUID = 1;

  public Dense (double[][] A)
  { super(A); }

  public Dense (double[][] A, int m, int n) 
  { super(A, m, n); }

  public Dense (double[] vals, int m) 
  { super(vals, m); }

  public Dense (int m, int n)
  { super(m, n); }

  public Dense (int m, int n, double s)
  { super(m, n, s); }

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

    LowRank lr = new LowRank (m, n, r);
    lr.setU(Q.times(svd_.getV()));
    lr.setS(svd_.getS());
    lr.setVT(V.times(svd_.getU()));

    return lr;
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
  public Hierarchical toHierarchical (int m, int n) {
    Hierarchical h = new Hierarchical(m, n);
    int i0 = 0;
    int step_i = (getRowDimension() - m + 1) / m, step_j = (getColumnDimension() - n + 1) / n;

    for (int i = 0; i < m; i++) {
      int i1 = i0 + step_i >= getRowDimension() ? getRowDimension() - 1 : i0 + step_i, j0 = 0;
      for (int j = 0; j < n; j++) {
        int j1 = j0 + step_j >= getColumnDimension() ? getColumnDimension() - 1 : j0 + step_j;
        Dense d = new Dense(getMatrix(i0, i1, j0, j1).getArrayCopy());
        h.setElement(i, j, d);
        j0 = j1 + 1;
      }
      i0 = i1 + 1;
    }

    return h;
  }

  @Override
  public Hierarchical toHierarchical (int level, int m, int n)
  {
    Hierarchical h = toHierarchical(m, n);
    if (level > 1) {
      for (int i = 0; i < h.getNRowBlocks(); i++) {
        for (int j = 0; j < h.getNColumnBlocks(); j++) {
          h.setElement(i, j, h.getElement(i, j).toHierarchical(level - 1, m, n));
        }
      }
    }
    return h;
  }

  @Override
  public boolean testAdmis (Matrix row_basis, Matrix col_basis, double admis_cond) {
    double row_err = row_basis.times(row_basis.transpose()).times(this).minus(this).normF();
    double col_err = times(col_basis).times(col_basis.transpose()).minus(this).normF();
    double admis_ref = admis_cond * getColumnDimension() * getRowDimension();
    return row_err <= admis_ref && col_err <= admis_ref;
  }

  @Override
  public boolean equals (Block b) {
    double norm = this.minus(b.toDense()).normF() / getColumnDimension() / getRowDimension();
    return norm <= PsplHMatrixPack.epi; 
  }

  @Override
  public double getCompressionRatio() 
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

  public static Dense generateDense (int m, int n, int y_start, int x_start, PsplHMatrixPack.dataFunction func) {
    Dense d = new Dense(m, n);
    double data[][] = d.getArray();

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++)
      { data[i][j] = func.body(i, j, y_start, x_start); }
    }

    return d;
  }

  public Matrix[] projection (Matrix row_basis, Matrix col_basis) {
    Matrix row_projection = row_basis.transpose().times(this);
    Matrix row_nulled = minus(row_basis.times(row_projection));
    Matrix col_projection = col_basis.transpose().times(transpose());
    Matrix col_nulled = transpose().minus(col_basis.times(col_projection));

    SingularValueDecomposition svd_0 = row_nulled.svd(), svd_1 = col_nulled.svd();
    Matrix row_null_space = svd_0.getU().getMatrix(0, row_basis.getRowDimension() - 1, 0, getRowDimension() - row_basis.getColumnDimension() - 1);
    Matrix col_null_space = svd_1.getU().getMatrix(0, col_basis.getRowDimension() - 1, 0, getColumnDimension() - col_basis.getColumnDimension() - 1);

    Matrix list[] = projection(row_basis, row_null_space, col_basis, col_null_space);

    return new Matrix[] {list[0], list[1], list[2], list[3], row_null_space, col_null_space};
  }

  public Matrix[] projection (Matrix row_basis, Matrix row_null_space, Matrix col_basis, Matrix col_null_space) {
    Matrix list[] = new Matrix[4];
    list[0] = row_basis.transpose().times(this).times(col_basis);
    list[1] = row_null_space.transpose().times(this).times(col_basis);
    list[2] = row_basis.transpose().times(this).times(col_null_space);
    list[3] = row_null_space.transpose().times(this).times(col_null_space);
    return list;
  }


}
