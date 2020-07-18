import java.io.*;
import Jama.Matrix;

public class Hierarchical implements Block {

  private Block[][] e;
  private LowRankBasic accm = null;

  public Hierarchical (int m, int n)
  { e = new Block[m][n]; }

  public Hierarchical (int m, int n, int nleaf, int partStrat, double admis, int yStart, int xStart, PsplHMatrixPack.DataFunction func, double[] rand) {
    int mBlock = m / partStrat;
    int mRemain = m - (partStrat - 1) * mBlock;
    int nBlock = n / partStrat;
    int nRemain = n - (partStrat - 1) * nBlock;

    e = new Block[partStrat][partStrat];

    for (int i = 0; i < partStrat; i++) {
      int mE = i == partStrat - 1 ? mRemain : mBlock;
      int yE = yStart + mBlock * i;

      for (int j = 0; j < partStrat; j++) {
        int nE = j == partStrat - 1 ? nRemain : nBlock;
        int xE = xStart + nBlock * j;

        boolean admisible = Integer.max(mE, nE) <= admis * Math.abs(xE - yE);

        if (admisible)
        { e[i][j] = new Dense(mE, nE, yE, xE, func, rand).toLowRank(); }
        else if (mE <= nleaf || nE <= nleaf)
        { e[i][j] = new Dense(mE, nE, yE, xE, func, rand); }
        else
        { e[i][j] = new Hierarchical(mE, nE, nleaf, partStrat, admis, yE, xE, func, rand); }

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
      int i1 = 0;
      int j0 = 0;
      for (int j = 0; j < getNColumnBlocks(); j++) {
        Dense x = e[i][j].toDense(); 
        int j1 = j0 + x.getColumnDimension() - 1;
        i1 = i0 + x.getRowDimension() - 1;
        d.setMatrix(i0, i1, j0, j1, x);
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
    double norm = 0.;
    int i0 = 0;
    for (int i = 0; i < getNRowBlocks(); i++) {
      int i1 = 0;
      int j0 = 0;
      for (int j = 0; j < getNColumnBlocks(); j++) {
        int l = e[i][j].getRowDimension();
        int n = e[i][j].getColumnDimension();
        int j1 = j0 + n - 1;
        i1 = i0 + l - 1;
        norm += e[i][j].compare(m.getMatrix(i0, i1, j0, j1));
        j0 = j1 + 1;
      }
      i0 = i1 + 1;
    }
    return norm;
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
    PsplHMatrixPack.errorOut("This method shouldn't be used in non-shared H-matrix.");
    return -1;
  }

  @Override
  public String structure () {
    StringBuilder s = new StringBuilder("H " + Integer.toString(getNRowBlocks()) + " " + Integer.toString(getNColumnBlocks()) + "\n");

    for (int i = 0; i < getNRowBlocks(); i++) {
      for (int j = 0; j < getNColumnBlocks(); j++)
      { s.append(e[i][j].structure()); }
    }

    return s.toString();
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
  public Block trsm (Block b, boolean lower) {
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

  @Override
  public Block copyBlock () {
    return null;
  }

  public Block getElement (int m, int n)
  { return e[m][n]; }

  public void setElement (int m, int n, Block b) {
    if (m < getNRowBlocks() && n < getNColumnBlocks())
    { e[m][n] = b; }
  }


}
