import Jama.Matrix;

public class UniformBLR {

  private Matrix row_basis[];
  private Matrix col_basis[];
  private Block e[][];

  public UniformBLR (int m, int n, int block_size, int y_start, int x_start, int rank, double admis, PsplHMatrixPack.dataFunction func) {
    int rows = (m + block_size - 1) / block_size, last_row = m - (rows - 1) * block_size;
    int cols = (n + block_size - 1) / block_size, last_col = n - (cols - 1) * block_size;
    e = new Block[rows][cols];
    row_basis = new Matrix[rows];
    col_basis = new Matrix[cols];

    for (int j = 0; j < cols; j++) {
      int n_e = j == cols - 1 ? last_col : block_size;
      int x_e = x_start + j * block_size;
      col_basis[j] = Dense.getBasisVT(x_e, n_e, rank, admis, func);
    }

    for (int i = 0; i < rows; i++) {
      int m_e = i == rows - 1 ? last_row : block_size;
      int y_e = y_start + i * block_size;
      row_basis[i] = Dense.getBasisU(y_e, m_e, rank, admis, func);
      
      for (int j = 0; j < cols; j++) {
        int n_e = j == cols - 1 ? last_col : block_size;
        int x_e = x_start + j * block_size;

        boolean admisible = Integer.max(m_e, n_e) <= admis * Math.abs(x_e - y_e);

        if (admisible)
        { e[i][j] = new Dense(m_e, n_e, y_e, x_e, func).toLowRank_fromBasis(row_basis[i], col_basis[j]); }
        else
        { e[i][j] = new Dense(m_e, n_e, y_e, x_e, func); }
      }
    }

  }

  public int getNRowBlocks()
  { return e.length; }

  public int getNColumnBlocks()
  { return e[0].length; }

  public int getRowDimension() {
    int accum = 0;
    for (int i = 0; i < getNRowBlocks(); i++)
    { accum += e[i][0].getRowDimension(); }
    return accum;
  }

  public int getColumnDimension() {
    int accum = 0;
    for (int i = 0; i < getNColumnBlocks(); i++)
    { accum += e[0][i].getColumnDimension(); }
    return accum;
  }

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



}