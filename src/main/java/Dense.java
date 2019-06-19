
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
  public LowRank toLowRank()
  {
    int m = getRowDimension(), n = getColumnDimension();
    int step = n > 32 ? 8 : n / 4, r = 0;

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
      double norm = A.normF() / (m * r);
      approx = norm <= 1.e-10;

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

  @Override
  public Hierarchical toHierarchical (int m, int n)
  {
    Hierarchical h = new Hierarchical(m, n);
    int i0 = 0;
    int step_i = getRowDimension() / m - 1, step_j = getColumnDimension() / n - 1;

    for (int i = 0; i < m; i++)
    {
      int i1 = i0 + step_i >= getRowDimension() ? getRowDimension() - 1 : i0 + step_i, j0 = 0;
      for (int j = 0; j < n; j++)
      {
        int j1 = j0 + step_j >= getColumnDimension() ? getColumnDimension() - 1 : j0 + step_j;
        Dense d = new Dense(i1 - i0 + 1, j1 - j0 + 1); 
        d.plusEquals(getMatrix(i0, i1, j0, j1));
        h.setElement(i, j, d);
        j0 = j1 + 1;
      }
      i0 = i1 + 1;
    }

    return h;
  }

  @Override
  public boolean equals (Block b) 
  {
    double norm = this.minus(b.toDense()).normF() / getColumnDimension() / getRowDimension();
    return norm <= 1.e-10; 
  }

  @Override
  public String structure ()
  { return "D " + Integer.toString(getRowDimension()) + " " + Integer.toString(getColumnDimension()) + "\n"; }

  @Override
  public void loadBinary (InputStream stream) throws IOException
  {
    int m = getRowDimension(), n = getColumnDimension();
    byte data[] = stream.readNBytes(8 * m * n);
    double data_ptr[][] = getArray();

    for (int i = 0; i < m; i++)
    {
      for (int j = 0; j < n; j++)
      { data_ptr[i][j] = ByteBuffer.wrap(data).getDouble((i * n + j) * 8); }
    }
  }

  @Override
  public void writeBinary (OutputStream stream) throws IOException
  {
    int m = getRowDimension(), n = getColumnDimension();
    byte data[] = new byte[8 * m * n];
    double data_ptr[][] = getArray();

    for (int i = 0; i < m; i++)
    {
      for (int j = 0; j < n; j++)
      { ByteBuffer.wrap(data).putDouble((i * n + j) * 8, data_ptr[i][j]); }
    }

    stream.write(data);
  }

  @Override
  public void writeToFile (String name) throws IOException
  {
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

  public static Dense readFromFile (String name) throws IOException
  {
    BufferedReader reader = new BufferedReader(new FileReader("bin/" + name + ".struct"));
    String str = reader.readLine();
    reader.close();

    if (str.startsWith("H"))
    {
      Hierarchical h = Hierarchical.readFromFile(name);
      return h.toDense();
    }
    else if (str.startsWith("LR"))
    {
      LowRank lr = LowRank.readFromFile(name);
      return lr.toDense();
    }
    else if (str.startsWith("D"))
    {
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


}
