
import java.io.*;

public class H2Matrix implements Block {

  private ClusterBasis row_basis;
  private ClusterBasis col_basis;
  private Block e[][];
  private int x_start = 0;
  private int y_start = 0;


  public H2Matrix (int m, int n, int nleaf, int part_strat, int rank, double admis, int y_start, int x_start, PsplHMatrixPack.dataFunction func) {
    row_basis = new ClusterBasis(y_start, m, true, nleaf, part_strat, rank, admis, func);
    col_basis = new ClusterBasis(x_start, n, false, nleaf, part_strat, rank, admis, func);
    e = new Block[part_strat][part_strat];

    ClusterBasis[] row_basis_lower = row_basis.getChildren(), col_basis_lower = col_basis.getChildren();

    int m_block = m / part_strat, m_remain = m - (part_strat - 1) * m_block;
    int n_block = n / part_strat, n_remain = n - (part_strat - 1) * n_block;

    for (int i = 0; i < part_strat; i++) {
      int m_e = i == part_strat - 1 ? m_remain : m_block;
      int y_e = y_start + m_block * i;
      ClusterBasis b_i = row_basis_lower[i];

      for (int j = 0; j < part_strat; j++) {
        int n_e = j == part_strat - 1 ? n_remain : n_block;
        int x_e = x_start + n_block * j;
        ClusterBasis b_j = col_basis_lower[j];

        boolean admisible = Integer.max(m_e, n_e) <= admis * Math.abs(x_e - y_e);

        if (admisible)
        { e[i][j] = new Dense(m_e, n_e, y_e, x_e, func).toLowRank_fromBasis(b_i, b_j); }
        else if (b_i.noChildren() || b_j.noChildren())
        { e[i][j] = new Dense(m_e, n_e, y_e, x_e, func); }
        else
        { e[i][j] = new H2Matrix(b_i, b_j, admis, func); }

      }
    }

    //row_basis.convertReducedStorageForm();
    //col_basis.convertReducedStorageForm();
  }

  public H2Matrix (ClusterBasis row_basis, ClusterBasis col_basis, double admis, PsplHMatrixPack.dataFunction func) {
    this.row_basis = row_basis;
    this.col_basis = col_basis;
    int y_start = row_basis.getStart(), x_start = col_basis.getStart();
    int m = row_basis.getDimension(), n = col_basis.getDimension();
    int part_strat = row_basis.getPartStrat();
    e = new Block[part_strat][part_strat];

    ClusterBasis[] row_basis_lower = row_basis.getChildren(), col_basis_lower = col_basis.getChildren();

    int m_block = m / part_strat, m_remain = m - (part_strat - 1) * m_block;
    int n_block = n / part_strat, n_remain = n - (part_strat - 1) * n_block;

    for (int i = 0; i < part_strat; i++) {
      int m_e = i == part_strat - 1 ? m_remain : m_block;
      int y_e = y_start + m_block * i;
      ClusterBasis b_i = row_basis_lower[i];

      for (int j = 0; j < part_strat; j++) {
        int n_e = j == part_strat - 1 ? n_remain : n_block;
        int x_e = x_start + n_block * j;
        ClusterBasis b_j = col_basis_lower[j];

        boolean admisible = Integer.max(m_e, n_e) <= admis * Math.abs(x_e - y_e);

        if (admisible)
        { e[i][j] = new Dense(m_e, n_e, y_e, x_e, func).toLowRank_fromBasis(b_i, b_j); }
        else if (b_i.noChildren() || b_j.noChildren())
        { e[i][j] = new Dense(m_e, n_e, y_e, x_e, func); }
        else
        { e[i][j] = new H2Matrix(b_i, b_j, admis, func); }

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
  public boolean equals (Block b) {
    double norm = this.toDense().minus(b.toDense()).normF() / getRowDimension() / getColumnDimension();
    return norm < PsplHMatrixPack.epi;
  }

  @Override
  public double getCompressionRatio () {
    double compress = 0.;

    for (int i = 0; i < getNRowBlocks(); i++) {
      for (int j = 0; j < getNColumnBlocks(); j++)
      { compress += e[i][j].getCompressionRatio_NoBasis() * e[i][j].getRowDimension() * e[i][j].getColumnDimension(); }
    }

    compress += row_basis.size() + col_basis.size();
    return compress / getColumnDimension() / getRowDimension();
  }

  @Override
  public double getCompressionRatio_NoBasis () {
    double compress = 0.;

    for (int i = 0; i < getNRowBlocks(); i++) {
      for (int j = 0; j < getNColumnBlocks(); j++)
      { compress += e[i][j].getCompressionRatio_NoBasis() * e[i][j].getRowDimension() * e[i][j].getColumnDimension(); }
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



}