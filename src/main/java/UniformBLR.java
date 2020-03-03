

public class UniformBLR {

  private ClusterBasisU row_basis;
  private ClusterBasisV col_basis;
  private Block e[][];

  public UniformBLR (int m, int n, int block_size, int y_start, int x_start, double admis, PsplHMatrixPack.dataFunction func) {
    int rows = (m + block_size - 1) / block_size, last_row = m - rows * block_size;
    int cols = (n + block_size - 1) / block_size, last_col = n - cols * block_size;
    e = new Block[rows][cols];

    for (int i = 0; i < rows; i++) {
      int rows_i = i == rows - 1 ? last_row : block_size;
      int y_start_i = y_start + i * block_size;
      
      for (int j = 0; j < cols; j++) {
        int cols_j = j == cols - 1 ? last_col : block_size;
        int x_start_j = x_start + j * block_size;
        e[i][j] = new Dense(rows_i, cols_j, y_start_i, x_start_j, func);
      }
    }

  }


}