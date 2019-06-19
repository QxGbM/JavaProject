import java.io.*;

public class Hierarchical implements Block {

  private Block e[][];

  public Hierarchical (int m, int n)
  { e = new Block[m][n]; }

  public int getNRowBlocks()
  { return e.length; }

  public int getNColumnBlocks()
  { return e[0].length; }

  @Override
  public int getRowDimension() 
  {
    int accum = 0;
    for (int i = 0; i < getNRowBlocks(); i++)
    { accum += e[i][0].getRowDimension(); }
    return accum;
  }

  @Override
  public int getColumnDimension() 
  {
    int accum = 0;
    for (int i = 0; i < getNColumnBlocks(); i++)
    { accum += e[0][i].getColumnDimension(); }
    return accum;
  }

  @Override
  public Block_t getType() 
  { return Block_t.HIERARCHICAL; }

  @Override
  public Dense toDense() 
  {
    Dense d = new Dense(getRowDimension(), getColumnDimension());
    int i0 = 0;

    for (int i = 0; i < getNRowBlocks(); i++)
    {
      int i1 = 0, j0 = 0;
      for (int j = 0; j < getNColumnBlocks(); j++)
      {
        Dense X = e[i][j].toDense(); 
        int j1 = j0 + X.getColumnDimension() - 1;
        i1 = i0 + X.getRowDimension() - 1;
        d.setMatrix(i0, i1, j0, j1, X);
        j0 = j1 + 1;
      }
      i0 = i1 + 1;
    }

    return d;
  }

  @Override
  public LowRank toLowRank() 
  { return toDense().toLowRank(); }

  @Override
  public Hierarchical toHierarchical (int m, int n) 
  {
    if (m == getNRowBlocks() && n == getNColumnBlocks())
    { return this; }
    else
    { return toDense().toHierarchical(m, n); }
  }

  @Override
  public boolean equals (Block b) 
  {
    double norm = this.toDense().minus(b.toDense()).normF() / getRowDimension() / getColumnDimension();
    return norm < 1.e-10;
  }

  @Override
  public String structure ()
  {
    String s = "H " + Integer.toString(getNRowBlocks()) + " " + Integer.toString(getNColumnBlocks()) + "\n";

    for (int i = 0; i < getNRowBlocks(); i++)
    {
      for (int j = 0; j < getNColumnBlocks(); j++)
      { s += e[i][j].structure(); }
    }

    return s;
  }

  @Override
  public void writeBinary (OutputStream stream) throws IOException
  {
    for (int i = 0; i < getNRowBlocks(); i++)
    {
      for (int j = 0; j < getNColumnBlocks(); j++)
      { e[i][j].writeBinary(stream); }
    }
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
  {
    for (int i = 0; i < getNRowBlocks(); i++)
    {
      for (int j = 0; j < getNColumnBlocks(); j++)
      { e[i][j].print(w, d); }
    }
  }

  public void setElement (int m, int n, Block b)
  {
    if (m < getNRowBlocks() && n < getNColumnBlocks())
    { e[m][n] = b; }
  }

  public static Hierarchical readFromFile (String name) throws IOException
  {
    return null;
  }
	
	
}
