import java.io.*;
import Jama.Matrix;

public class Hierarchical implements Block {

  private Block e[][];
  private int x_start = 0;
  private int y_start = 0;
  private LowRankBasic accm = null;

  public Hierarchical (int m, int n)
  { e = new Block[m][n]; }

  public Hierarchical (int m, int n, int nleaf, int part_strat, double admis, int y_start, int x_start, PsplHMatrixPack.dataFunction func) {
    int m_block = m / part_strat, m_remain = m - (part_strat - 1) * m_block;
    int n_block = n / part_strat, n_remain = n - (part_strat - 1) * n_block;

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
  { return null; }
  
  @Override
  public LowRankBasic toLowRankBasic () { 
    return null;
  }

  @Override
  public Hierarchical castHierarchical() {
    return this;
  }

  @Override
  public H2Matrix castH2Matrix() {
    return null;
  }

  @Override
  public void setAccumulator (LowRankBasic accm) {
    this.accm = accm;
  }

  @Override
  public LowRankBasic getAccumulator() {
    return accm;
  }

  @Override
  public boolean equals (Block b) {
    double norm = compare(b.toDense());
    return norm <= PsplHMatrixPack.EPI; 
  }

  @Override
  public double compare (Matrix m) {
    return this.toDense().minus(m).normF() / getColumnDimension() / getRowDimension();
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
  public double getCompressionRatio_NoBasis () {
    System.out.println("This method shouldn't be used in non-shared H-matrix.");
    return -1;
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
  public Block LU () {
    return this;
  }

  @Override
  public Block triangularSolve (Block b, boolean up_low) {
    return this;
  }

  @Override
  public Block GEMatrixMult (Block a, Block b, double alpha, double beta) {
    System.exit(1);
    return null;
  }

  @Override
  public Block GEMatrixMult (Block a, Block b, double alpha, double beta, ClusterBasisProduct X, ClusterBasisProduct Y, ClusterBasisProduct Z, H2Approx Sa, H2Approx Sb, H2Approx Sc) {
    System.exit(1);
    return null;
  }


  @Override
  public Block plusEquals (Block b) {
    System.exit(1);
    return null;
  }

  @Override
  public Block scalarEquals (double s) {
    System.exit(1);
    return null;
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
      return null;
    }
    else if (str.startsWith("LR")) {
      reader.close();
      return null;
    }
    else if (str.startsWith("H")) {
      Hierarchical h = new Hierarchical(m, n);

      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++)
        { h.e[i][j] = Block.readStructureFromFile(reader); }
      }
      return h;
    }
    else { 
      reader.close();
      return null;
    }  
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


}
