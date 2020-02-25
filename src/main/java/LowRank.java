
import java.io.*;
import java.nio.ByteBuffer;

import Jama.Matrix;

public class LowRank implements Block {
		
  private Matrix U, S, VT;
  private int x_start = 0;
  private int y_start = 0;

  public LowRank (int m, int n, int r) {
    U = new Matrix(m, r);
    S = new Matrix(r, r);
    VT = new Matrix(n, r);
  }

  public LowRank (Matrix row_basis, Matrix d) {
    U = row_basis; VT = d;
    S = Matrix.identity(row_basis.getColumnDimension(), d.getColumnDimension());
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
  { return U.getRowDimension(); }

  @Override
  public int getColumnDimension() 
  { return VT.getRowDimension(); }

  public int getRank()
  { return S.getRowDimension(); }

  @Override
  public Block_t getType() 
  { return Block_t.LOW_RANK; }

  @Override
  public Dense toDense() {
    Dense d = new Dense(getRowDimension(), getColumnDimension());
    d.setClusterStart(x_start, y_start);
    d.plusEquals(U.times(S).times(VT.transpose()));
    return d;
  }

  @Override
  public LowRank toLowRank() 
  { return this; }

  @Override
  public Hierarchical toHierarchical (int m, int n) {
    Hierarchical h = new Hierarchical(m, n);
    h.setClusterStart(x_start, y_start);
    int i0 = 0, r = getRank();
    int step_i = (getRowDimension() - m + 1) / m, step_j = (getColumnDimension() - n + 1) / n;

    for (int i = 0; i < m; i++) {
      int i1 = i0 + step_i >= getRowDimension() ? getRowDimension() - 1 : i0 + step_i, j0 = 0;
      for (int j = 0; j < n; j++) {
        int j1 = j0 + step_j >= getColumnDimension() ? getColumnDimension() - 1 : j0 + step_j;

        LowRank lr = new LowRank(i1 - i0 + 1, j1 - j0 + 1, r);
        lr.setS(S);
        lr.setU(U.getMatrix(i0, i1, 0, getRank() - 1));
        lr.setVT(VT.getMatrix(j0, j1, 0, getRank() - 1));

        h.setElement(i, j, lr);
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
    h.setClusterStart(x_start, y_start);
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
    double row_err = row_basis.times(row_basis.transpose()).times(U).minus(U).normF();
    double col_err = col_basis.times(col_basis.transpose()).times(VT).minus(VT).normF();
    double admis_ref = admis_cond * getColumnDimension() * getRowDimension();
    return row_err <= admis_ref && col_err <= admis_ref;
  }

  @Override
  public boolean equals (Block b) {
    double norm = this.toDense().minus(b.toDense()).normF() / getRowDimension() / getColumnDimension();
    return norm < PsplHMatrixPack.epi;
  }

  @Override
  public double getCompressionRatio ()
  { return (double) getRank() * (getColumnDimension() + getRowDimension()) / (getColumnDimension() * getRowDimension()); }

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

    S = Matrix.identity(r, r);

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
        ByteBuffer.wrap(data).putDouble(0, data_ptr[i][j] * S.get(j, j));
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
  { U.print(w, d); S.print(w, d); VT.print(w, d); }

  @Override
  public void LU () {
    System.out.println("error LU on LR");
  }

  @Override
  public void triangularSolve (Block b, boolean up_low) {

  }

  @Override
  public void GEMatrixMult (Block a, Block b, double alpha, double beta) {

  }

  public void setU (Matrix U)
  { this.U = new Matrix(U.getArray()); }

  public void setS (Matrix S)
  { this.S = new Matrix(S.getArray()); }

  public void setVT (Matrix VT)
  { this.VT = new Matrix(VT.getArray()); }

  public Matrix getU ()
  { return U; }

  public Matrix getS ()
  { return S; }
  
  public Matrix getVT ()
  { return VT; }

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
