
import Jama.Matrix;

public class ClusterBasis {

  private Matrix elements[];
  private ClusterBasis children[];
  
  public ClusterBasis (int n, int dim) {
    elements = new Matrix[n];
    children = null;
    for (int i = 0; i < n; i++)
    { elements[i] = new Matrix(dim, dim); }
  }

  public ClusterBasis (int n, int dim, int n_child) {
    elements = new Matrix[n];
    children = new ClusterBasis[n_child];
    for (int i = 0; i < n; i++)
    { elements[i] = new Matrix(dim, dim); }
  }

  public void setElement (int index, Matrix m) {
    elements[index] = m.copy();
  }

  public void setChildren (int index, ClusterBasis b) {
    children[index] = b;
  }

  public Matrix getBasis (int index) {
    return elements[index];
  }

  public Dense applyLeft (Dense d) {
    int row = 0, e = 0; 
    Dense result_d = new Dense(d.getColumnDimension(), d.getRowDimension());
    double[][] data = result_d.getArray();
    while (row < d.getColumnDimension()) {
      int row_new = row + elements[e].getColumnDimension() - 1;
      Matrix m = elements[e].times(d.getMatrix(row, row_new, 0, d.getRowDimension() - 1));
      for (int i = row; i <= row_new; i++) {
        for (int j = 0; j < m.getColumnDimension(); j++)
        { data[i][j] = m.get(i - row, j); }
      }
      row = row_new + 1; e++;
    }
    return result_d;
  }

  public Dense solveLeft (Dense d) {
    int row = 0, e = 0; 
    Dense result_d = new Dense(d.getColumnDimension(), d.getRowDimension());
    double[][] data = result_d.getArray();
    while (row < d.getColumnDimension()) {
      int row_new = row + elements[e].getColumnDimension() - 1;
      Matrix m = elements[e].transpose().times(d.getMatrix(row, row_new, 0, d.getRowDimension() - 1));
      for (int i = row; i <= row_new; i++) {
        for (int j = 0; j < m.getColumnDimension(); j++)
        { data[i][j] = m.get(i - row, j); }
      }
      row = row_new + 1; e++;
    }
    return result_d;
  }

  public void print (int w, int d) {
    for (int i = 0; i < elements.length; i++)
    {
      System.out.println("B"+i);
      elements[i].print(w, d);
    }
  }

  public static Matrix[] checkerlize (Matrix m, Matrix ql, Matrix qr, int[] joints) {
    Matrix[] list = new Matrix[joints.length + 1];
    Matrix temp = m.copy();
    for (int i = 0; i <= joints.length; i++) {
      int start = i == 0 ? 0 : joints[i - 1], end = i == joints.length ? m.getRowDimension() - 1 : joints[i] - 1;
      Matrix left = ql.getMatrix(0, ql.getRowDimension() - 1, start, end), right = qr.getMatrix(start, end, 0, qr.getColumnDimension() - 1);
      list[i] = left.transpose().times(temp).times(right.transpose());
      temp.minusEquals(left.times(list[i]).times(right));
      System.out.println("rank" + temp.rank());
    }
    return list;
  }

  public static Matrix uncheckerlize (Matrix[] checkers, Matrix ql, Matrix qr) {
    Matrix result_m = new Matrix(ql.getRowDimension(), qr.getColumnDimension());
    int start = 0, end;
    for (int i = 0; i < checkers.length; i++) {
      end = start + checkers[i].getRowDimension() - 1;
      Matrix left = ql.getMatrix(0, ql.getRowDimension() - 1, start, end), right = qr.getMatrix(start, end, 0, qr.getColumnDimension() - 1);
      result_m.plusEquals(left.times(checkers[i]).times(right));
      start = end + 1;
    }
    return result_m;
  }

}