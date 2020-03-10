
import java.io.*;
import java.nio.ByteBuffer;

import Jama.Matrix;

public class LowRank implements Block {
		
  private Matrix S;
  private ClusterBasis U, VT;
  private int x_start = 0;
  private int y_start = 0;

  public LowRank (int m, int n, int r) {
    U = new ClusterBasis(m, r, true);
    S = new Matrix(r, r);
    VT = new ClusterBasis(n, r, false);
  }

  public LowRank (ClusterBasis U, Matrix S, ClusterBasis VT) {
    this.U = U;
    this.VT = VT;

    if (U.getDimension() == S.getRowDimension() && VT.getDimension() == S.getColumnDimension())
    { this.S = U.toMatrix().transpose().times(S).times(VT.toMatrix()); }
    else if (U.getRank() == S.getRowDimension() && VT.getRank() == S.getColumnDimension())
    { this.S = S; }
    else
    { System.out.print("Invalid Low-Rank Construction."); System.exit(-1); }
  }

  public LowRank (Matrix U, Matrix S, Matrix VT) {
    ClusterBasis row_b = new ClusterBasis(U, true);
    ClusterBasis col_b = new ClusterBasis(VT, false);

    this.U = row_b;
    this.VT = col_b;

    if (U.getRowDimension() == S.getRowDimension() && VT.getRowDimension() == S.getColumnDimension())
    { this.S = U.transpose().times(S).times(VT); }
    else if (U.getColumnDimension() == S.getRowDimension() && VT.getColumnDimension() == S.getColumnDimension())
    { this.S = S; }
    else
    { System.out.print("Invalid Low-Rank Construction."); System.exit(-1); }
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
  { return S.getRowDimension(); }

  @Override
  public Block_t getType() 
  { return Block_t.LOW_RANK; }

  @Override
  public Dense toDense() {
    Matrix m = U.toMatrix().times(S).times(VT.toMatrix().transpose());
    return new Dense(m.getArray());
  }

  @Override
  public LowRank toLowRank() 
  { return this; }

  @Override
  public boolean equals (Block b) {
    double norm = this.toDense().minus(b.toDense()).normF() / getRowDimension() / getColumnDimension();
    return norm < PsplHMatrixPack.epi;
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
  public void LU () {
    System.out.println("error LU on LR");
  }

  @Override
  public void triangularSolve (Block b, boolean up_low) {

  }

  @Override
  public void GEMatrixMult (Block a, Block b, double alpha, double beta) {

  }

  public ClusterBasis getU ()
  { return U; }

  public Matrix getS ()
  { return S; }
  
  public ClusterBasis getVT ()
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
