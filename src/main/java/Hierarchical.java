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
  public int getRowDimension() {
    int accum = 0;
    for (int i = 0; i < getNRowBlocks(); i++)
    { accum += e[i][0].getRowDimension(); }
    return accum;
  }

  @Override
  public int getColumnDimension() {
    int accum = 0;
    for (int i = 0; i < getNColumnBlocks(); i++)
    { accum += e[0][i].getColumnDimension(); }
    return accum;
  }

  @Override
  public Block_t getType() 
  { return Block_t.HIERARCHICAL; }

  @Override
  public Dense toDense() {
    Dense d = new Dense(getRowDimension(), getColumnDimension());
    int i0 = 0;

    for (int i = 0; i < getNRowBlocks(); i++) {
      int i1 = 0, j0 = 0;
      for (int j = 0; j < getNColumnBlocks(); j++) {
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
  public Hierarchical toHierarchical (int m, int n) {
    if (m == getNRowBlocks() && n == getNColumnBlocks())
    { return this; }
    else
    { return toDense().toHierarchical(m, n); }
  }

  @Override
  public boolean equals (Block b) {
    double norm = this.toDense().minus(b.toDense()).normF() / getRowDimension() / getColumnDimension();
    return norm < PsplHMatrixPack.epi;
  }

  @Override
  public double getCompressionRatio () {
    double compress = 0.;

    for (int i = 0; i < getNRowBlocks(); i++) {
      for (int j = 0; j < getNColumnBlocks(); j++)
      { compress += e[i][j].getCompressionRatio() * e[i][j].getRowDimension() * e[i][j].getColumnDimension(); }
    }

    return compress / getColumnDimension() / getRowDimension();
  }

  @Override
  public String structure () {
    String s = "H " + Integer.toString(getNRowBlocks()) + " " + Integer.toString(getNColumnBlocks()) + "\n";

    for (int i = 0; i < getNRowBlocks(); i++) {
      for (int j = 0; j < getNColumnBlocks(); j++)
      { s += e[i][j].structure(); }
    }

    return s;
  }

  @Override
  public void loadBinary (InputStream stream) throws IOException {
    for (int i = 0; i < getNRowBlocks(); i++) {
      for (int j = 0; j < getNColumnBlocks(); j++)
      { e[i][j].loadBinary(stream); }
    }
  }

  @Override
  public void writeBinary (OutputStream stream) throws IOException {
    for (int i = 0; i < getNRowBlocks(); i++) {
      for (int j = 0; j < getNColumnBlocks(); j++)
      { e[i][j].writeBinary(stream); }
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
  public void print (int w, int d) {
    for (int i = 0; i < getNRowBlocks(); i++) {
      for (int j = 0; j < getNColumnBlocks(); j++)
      { e[i][j].print(w, d); }
    }
  }

  public void setElement (int m, int n, Block b) {
    if (m < getNRowBlocks() && n < getNColumnBlocks())
    { e[m][n] = b; }
  }

  public static Hierarchical readStructureFromFile (BufferedReader reader) throws IOException {
    String str = reader.readLine();
    String[] args = str.split("\\s+");
    int m = Integer.parseInt(args[1]), n = Integer.parseInt(args[2]);

    if (str.startsWith("D")) {
      reader.close();
      Dense d = new Dense(m, n);
      return d.toHierarchical(1, 1);
    }
    else if (str.startsWith("LR")) {
      reader.close();
      int r = Integer.parseInt(args[3]);
      LowRank lr = new LowRank(m, n, r);
      return lr.toHierarchical(1, 1);
    }
    else if (str.startsWith("H")) {
      Hierarchical h = new Hierarchical(m, n);

      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++)
        { h.e[i][j] = Block.readStructureFromFile(reader); }
      }
      return h;
    }
    else
    { return null; }  
  }

  public static Hierarchical readFromFile (String name) throws IOException {
    BufferedReader reader = new BufferedReader(new FileReader("bin/" + name + ".struct"));
    Hierarchical h = readStructureFromFile(reader);
    reader.close();

    BufferedInputStream stream = new BufferedInputStream(new FileInputStream("bin/" + name + ".bin"));
    h.loadBinary(stream);
    stream.close();
    return h;
  }

  public static Hierarchical buildHMatrix (int level, int nblocks, int dim, int admis, int y_start, int x_start, PsplHMatrixPack.dataFunction func) {
    Hierarchical h = new Hierarchical(nblocks, nblocks);
    int blockSize = dim / nblocks, old_x_start = x_start;

    for (int i = 0; i < nblocks; i++) {
      x_start = old_x_start;
      for (int j = 0; j < nblocks; j++) {
        int loc = Math.abs(y_start - x_start);
        boolean admisBlock = loc < admis + blockSize, admisLeaf = loc < (admis + 1) * blockSize;

        if (level > 0 && admisBlock)
        { h.e[i][j] = buildHMatrix (level - 1, nblocks, blockSize, admis, y_start, x_start, func); }
        else if (level <= 0 && admisLeaf)
        { h.e[i][j] = Dense.generateDense (blockSize, blockSize, y_start, x_start, func); }
        else {
          Dense d = Dense.generateDense (blockSize, blockSize, y_start, x_start, func);
          h.e[i][j] = d.toLowRank();
        }
        x_start += blockSize;
      }
      y_start += blockSize;
    }

    return h;
  }

}
