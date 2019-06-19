
import java.io.*;
import java.nio.ByteBuffer;

import Jama.Matrix;

public class LowRank implements Block {
		
  private Matrix U, S, VT;

  public LowRank (int m, int n, int r) 
  {
    U = new Matrix(m, r);
    S = new Matrix(r, r);
    VT = new Matrix(n, r);
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
  public Dense toDense() 
  {
    Dense d = new Dense(getRowDimension(), getColumnDimension());
    d.plusEquals(U.times(S).times(VT.transpose()));
    return d;
  }

  @Override
  public LowRank toLowRank() 
  { return this; }

  @Override
  public Hierarchical toHierarchical (int m, int n) 
  {
    Hierarchical h = new Hierarchical(m, n);
    int i0 = 0, r = getRank();
    int step_i = getRowDimension() / m - 1, step_j = getColumnDimension() / n - 1;

    for (int i = 0; i < m; i++)
    {
      int i1 = i0 + step_i >= getRowDimension() ? getRowDimension() - 1 : i0 + step_i, j0 = 0;
      for (int j = 0; j < n; j++)
      {
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
  public boolean equals (Block b) 
  {
    double norm = this.toDense().minus(b.toDense()).normF() / getRowDimension() / getColumnDimension();
    return norm < 1.e-10;
  }

  @Override
  public String structure ()
  { return "LR " + Integer.toString(getRowDimension()) + " " + Integer.toString(getColumnDimension()) + " " + Integer.toString(getRank()) + "\n"; }

  @Override
  public void writeBinary (OutputStream stream) throws IOException
  {
    int m = getRowDimension(), n = getColumnDimension(), r = getRank();
    byte data[] = new byte[8 * m * r];
    double data_ptr[][] = U.getArray();

    for (int i = 0; i < m; i++)
    {
      for (int j = 0; j < r; j++)
      { ByteBuffer.wrap(data).putDouble((i * r + j) * 8, data_ptr[i][j]); }
    }

    stream.write(data);

    data = new byte[8 * r * r];
    data_ptr = S.getArray();

    for (int i = 0; i < r; i++)
    {
      for (int j = 0; j < r; j++)
      { ByteBuffer.wrap(data).putDouble((i * r + j) * 8, data_ptr[i][j]); }
    }

    stream.write(data);

    data = new byte[8 * n * r];
    data_ptr = VT.getArray();

    for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < r; j++)
      { ByteBuffer.wrap(data).putDouble((i * r + j) * 8, data_ptr[i][j]); }
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
  { U.print(w, d); S.print(w, d); VT.print(w, d); }

  public void setU (Matrix U)
  { this.U.setMatrix(0, getRowDimension() - 1, 0, getRank() - 1, U); }

  public void setS (Matrix S)
  { this.S.setMatrix(0, getRank() - 1, 0, getRank() - 1, S); }

  public void setVT (Matrix VT)
  { this.VT.setMatrix(0, getColumnDimension() - 1, 0, getRank() - 1, VT); }

  public static LowRank readFromFile (String name) throws IOException
  {
    return null;
  }

}
