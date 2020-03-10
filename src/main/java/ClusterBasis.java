
import Jama.Matrix;

public class ClusterBasis {

  private Matrix basis;
  private ClusterBasis children[];
  private int xy_start;
  private boolean row_col;
  private boolean reducedStorageForm;

  public ClusterBasis (int m, int n, boolean row_col) {
    basis = new Matrix(m, n);
    children = null;
    xy_start = 0;
    this.row_col = row_col;
    reducedStorageForm = false;
  }

  public ClusterBasis (Matrix m, boolean row_col) {
    basis = new Matrix(m.getArray());
    children = null;
    xy_start = 0;
    this.row_col = row_col;
    reducedStorageForm = false;
  }

  public ClusterBasis (int xy_start, int mn, boolean row_col, int nleaf, int part_strat, int rank, double admis, PsplHMatrixPack.dataFunction func) {

    if (row_col)
    { basis = Dense.getBasisU(xy_start, mn, rank, admis, func); }
    else
    { basis = Dense.getBasisVT(xy_start, mn, rank, admis, func); }

    if (mn > nleaf) {
      int mn_block = mn / part_strat, mn_remain = mn - (part_strat - 1) * mn_block;
      children = new ClusterBasis[part_strat];
  
      for (int i = 0; i < part_strat; i++) {
        int mn_e = i == part_strat - 1 ? mn_remain : mn_block;
        int xy_e = xy_start + mn_block * i;
        children[i] = new ClusterBasis (xy_e, mn_e, row_col, nleaf, part_strat, rank, admis, func);
      }
    }
    else
    { children = null; }

    this.xy_start = xy_start;
    this.row_col = row_col;
    reducedStorageForm = false;
  }

  public int getDimension () {
    if (children != null) {
      int rows = 0; 
      for (int i = 0; i < children.length; i++)
      rows += children[i].getDimension();
      return rows;
    }
    else
    { return basis.getRowDimension(); }
  }

  public int getRank () {
    return basis.getColumnDimension();
  }

  public int size () {
    int sum = basis.getRowDimension() * basis.getColumnDimension();
    if (children != null) {
      for (int i = 0; i < children.length; i++)
      sum += children[i].size();
    }
    return sum;
  }

  public int getStart () {
    return xy_start;
  }

  public boolean noChildren () {
    return children == null;
  }

  public int getPartStrat () {
    return children.length;
  }

  public ClusterBasis[] getChildren () {
    return children;
  }

  public boolean getRow_Col () {
    return row_col;
  }

  public Matrix toMatrix() {
    if (!reducedStorageForm || children == null)
    { return basis; }
    else {
      int dim = 0; Matrix children_basis[] = new Matrix[children.length];
      for (int i = 0; i < children.length; i++) 
      { children_basis[i] = children[i].toMatrix(); dim += children_basis[i].getRowDimension(); }

      int start = 0; Matrix lower = new Matrix(dim, basis.getColumnDimension());
      for (int i = 0; i < children.length; i++)
      { lower.setMatrix(start, start += children_basis[i].getRowDimension() - 1, 0, lower.getColumnDimension() - 1, children_basis[i]); start++; }
      
      return lower.times(basis);
    }
  }

  public Matrix convertReducedStorageForm() {
    if (children == null)
    { reducedStorageForm = true; return basis; }
    else if (reducedStorageForm)
    { return toMatrix(); }
    else {
      int dim = 0; Matrix children_basis[] = new Matrix[children.length];
      for (int i = 0; i < children.length; i++) 
      { children_basis[i] = children[i].convertReducedStorageForm(); dim += children_basis[i].getRowDimension(); }

      int start = 0; Matrix lower = new Matrix(dim, basis.getColumnDimension());
      for (int i = 0; i < children.length; i++)
      { lower.setMatrix(start, start += children_basis[i].getRowDimension() - 1, 0, lower.getColumnDimension() - 1, children_basis[i]); start++; }

      Matrix temp = basis; basis = lower.inverse().times(temp);
      reducedStorageForm = true;
      return temp;
    }
  }


}