import java.io.*;
import Jama.Matrix;

public class Hierarchical implements Block {

  private Block e[][];
  private LowRankBasic accm = null;

  public Hierarchical (int m, int n)
  { e = new Block[m][n]; }

  public Hierarchical (int m, int n, int nleaf, int part_strat, double admis, int y_start, int x_start, PsplHMatrixPack.DataFunction func) {
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
  public double getCompressionRatioNoBasis () {
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
  public void print (int w, int d) {
    for (int i = 0; i < getNRowBlocks(); i++) {
      for (int j = 0; j < getNColumnBlocks(); j++)
      { e[i][j].print(w, d); }
    }
  }

  @Override
  public Block getrf () {
    return this;
  }

  @Override
  public Block trsm (Block b, boolean up_low) {
    return this;
  }

  @Override
  public Block gemm (Block a, Block b, double alpha, double beta) {
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

  @Override
  public Block times (Block b) {
    return null;
  }

  @Override
  public Block accum (LowRankBasic accm) {
    return null;
  }

  public Block getElement (int m, int n)
  { return e[m][n]; }

  public void setElement (int m, int n, Block b) {
    if (m < getNRowBlocks() && n < getNColumnBlocks())
    { e[m][n] = b; }
  }


}
