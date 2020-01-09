
import Jama.Matrix;

public class ClusterBasis {

  private Matrix basis;
  private ClusterBasis children[];
  
  public ClusterBasis (int m, int n) {
    basis = new Matrix(m, n);
    children = null;
  }

  public ClusterBasis (int m, int n, int n_child) {
    basis = new Matrix(m, n);
    children = new ClusterBasis[n_child];
  }

  public int getRowDimension () {
    if (children == null)
    return basis.getRowDimension();
    else { 
      int rows = 0; 
      for (int i = 0; i < children.length; i++)
      rows += children[i].getRowDimension();
      return rows;
    }
  }

  public int getColDimension () {
    if (children == null)
    return basis.getRowDimension();
    else 
    return children[0].getColDimension();
  }

  public void setBasis (Matrix m) {
    basis = m.copy();
  }

  public void setChildren (int index, ClusterBasis b) {
    children[index] = b;
  }

  public void applyLeft (Dense d) {
    d = new Dense(basis.times(d).getArray());
  }

  public void solveLeft (Dense d) {
    d = new Dense(basis.transpose().times(d).getArray());
  }

  public void applyRight (Dense d) {
    d = new Dense(d.times(basis).getArray());
  }

  public Matrix toMatrix () {
    if (children == null)
    { return basis; }
    else {
      Matrix result_b = new Matrix(getRowDimension(), getColDimension());
      int row = 0;
      for (int i = 0; i < children.length; i++) { 
        int rows = children[i].getRowDimension();
        Matrix data = basis.times(children[i].toMatrix());
        result_b.setMatrix(row, row + rows - 1, 0, getColDimension() - 1, data);
        row += rows;
      }
      return result_b;
    }
  }

  public void print (int w, int d) {
    toMatrix().print(w, d);
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