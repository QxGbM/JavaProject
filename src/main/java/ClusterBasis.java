
import Jama.Matrix;

public class ClusterBasis {

  private Matrix basis;
  private ClusterBasis children[];

  public ClusterBasis () {
    basis = null;
    children = null;
  }

  public ClusterBasis (Matrix m) {
    basis = m.copy();
    children = null;
  }

  public ClusterBasis (Dense d, int sample_rank) {
    Matrix[] rsv = d.rsvd(sample_rank);
    basis = rsv[0];
    children = null;
  }

  public int getRowDimension () {
    if (children != null) {
      int rows = 0; 
      for (int i = 0; i < children.length; i++)
      rows += children[i].getRowDimension();
      return rows;
    }
    else if (basis == null)
    return 0;
    else
    return basis.getRowDimension();
  }

  public int getColDimension () {
    if (children != null)
    return children[0].getColDimension();
    else if (basis == null)
    return 0;
    else
    return basis.getColumnDimension();
  }

  public boolean hasChildren () {
    return children != null;
  }

  public ClusterBasis[] getChildren() {
    return children;
  }

  public void setBasis (Matrix m) {
    basis = m.copy();
  }

  public void setChildren (ClusterBasis[] b) {
    children = new ClusterBasis[b.length];
    for (int i = 0; i < b.length; i++)
    { children[i] = b[i]; }
  }

  public Matrix convertTrans() {
    if (children != null && basis != null && getRowDimension() == basis.getRowDimension() && getColDimension() == basis.getColumnDimension()) {
      Matrix result_b = new Matrix (basis.getRowDimension(), basis.getColumnDimension());
      int row = 0;
      for (int i = 0; i < children.length; i++) { 
        int rows = children[i].getRowDimension();
        Matrix c_i = children[i].convertTrans();
        result_b.setMatrix(row, row + rows - 1, 0, basis.getColumnDimension() - 1, c_i);
        row += rows;
      }
      Matrix old = basis.copy();
      basis = old.transpose().times(result_b);
      return old;
    }
    else {
      return basis;
    }
  }

  public Matrix[] convertTrans_children() {
    Matrix[] list = new Matrix[children.length];
    for (int i = 0; i < children.length; i++)
    { list[i] = children[i].convertTrans(); }
    return list;
  }

  public Matrix toMatrix () {
    if (basis == null)
    { return null; }
    else if (children == null)
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

}