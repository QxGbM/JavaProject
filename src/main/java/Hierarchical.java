import java.io.*;
import Jama.Matrix;

public class Hierarchical implements Block {

  private Block e[][];
  private int x_start = 0;
  private int y_start = 0;

  public Hierarchical (int m, int n)
  { e = new Block[m][n]; }

  public Hierarchical (int m, int n, int nleaf, int part_strat, double admis, int y_start, int x_start, PsplHMatrixPack.dataFunction func) {
    int m_block = m / part_strat, n_block = n / part_strat;
    int m_remain = m - (part_strat - 1) * m_block, n_remain = n - (part_strat - 1) * n_block;

    e = new Block[part_strat][part_strat];

    for (int i = 0; i < part_strat; i++) {
      int m_e = i == part_strat - 1 ? m_remain : m_block;
      int y_e = y_start + m_block * i;

      for (int j = 0; j < part_strat; j++) {
        int n_e = j == part_strat - 1 ? n_remain : n_block;
        int x_e = x_start + n_block * j;

        boolean admisible = Integer.max(m_e, n_e) <= admis * Math.abs(x_e - y_e);

        if (admisible)
        { e[i][j] = new Dense(m_e, n_e, y_e, x_e, func).toLowRank(); }
        else if (m_e <= nleaf || n_e <= nleaf)
        { e[i][j] = new Dense(m_e, n_e, y_e, x_e, func); }
        else
        { e[i][j] = new Hierarchical(m_e, n_e, nleaf, part_strat, admis, y_e, x_e, func); }

      }
    }
  }

  public int getNRowBlocks()
  { return e.length; }

  public int getNColumnBlocks()
  { return e[0].length; }

  public Block[][] getElements ()
  { return e; }

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
    d.setClusterStart(x_start, y_start);
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
    return false;
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

  @Override
  public void LU () {
    
  }

  @Override
  public void triangularSolve (Block b, boolean up_low) {

  }

  @Override
  public void GEMatrixMult (Block a, Block b, double alpha, double beta) {

  }

  public Block getElement (int m, int n)
  { return e[m][n]; }

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

  public static Hierarchical buildHMatrix (int level, int nblocks, int nleaf, int nleaf_max, int admis, int y_start, int x_start, PsplHMatrixPack.dataFunction func) {
    Hierarchical h = new Hierarchical(nblocks, nblocks);
    h.setClusterStart(x_start, y_start);
    int old_x_start = x_start, blockSize = nleaf * (int) Math.pow(nblocks, level);

    for (int i = 0; i < nblocks; i++) {
      x_start = old_x_start;
      for (int j = 0; j < nblocks; j++) {
        int loc = Math.abs(y_start - x_start);
        boolean admisBlock = loc < admis + blockSize, admisLeaf = loc < (admis + 1) * nleaf;

        if (level > 0 && admisBlock) { 
          h.e[i][j] = buildHMatrix (level - 1, nblocks, nleaf, nleaf_max, admis, y_start, x_start, func); 
        }
        else if (level <= 0 && admisLeaf) { 
          h.e[i][j] = new Dense (nleaf, nleaf, y_start, x_start, func); 
        }
        else {
          Dense d = new Dense (blockSize, blockSize, y_start, x_start, func);
          LowRank lr = d.toLowRank();

          int e_level = 0, e_block = blockSize;
          while (e_block > nleaf_max)
          { e_level++; e_block /= nblocks; }

          if (e_level > 0)
          { h.e[i][j] = lr.toHierarchical(e_level, nblocks, nblocks); }
          else
          { h.e[i][j] = lr; }
        }
        x_start += blockSize;
      }
      y_start += blockSize;
    }

    return h;
  }

}
