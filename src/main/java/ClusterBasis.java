
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

  public ClusterBasis (Hierarchical h, int sample_rank) {
    basis = null;
    children = new ClusterBasis[h.getNRowBlocks()];

    children[0] = new ClusterBasis (h.getElement(0, 1).toDense(), sample_rank);

    for (int i = 1; i < h.getNRowBlocks(); i++)
    { children[1] = new ClusterBasis (h.getElement(i, 0).toDense(), sample_rank); }
  }

  public int getRowDimension () {
    if (basis == null)
    return 0;
    else if (children == null)
    return basis.getRowDimension();
    else { 
      int rows = 0; 
      for (int i = 0; i < children.length; i++)
      rows += children[i].getRowDimension();
      return rows;
    }
  }

  public int getColDimension () {
    if (basis == null)
    return 0;
    else if (children == null)
    return basis.getRowDimension();
    else 
    return children[0].getColDimension();
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
    children[i] = b[i];
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