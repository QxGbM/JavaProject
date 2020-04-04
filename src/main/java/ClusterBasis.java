
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

  public ClusterBasis (ClusterBasis cb, Matrix m) {
    basis = cb.basis.times(m);
    children = cb.children;
    xy_start = cb.xy_start;
    row_col = cb.row_col;
    reducedStorageForm = cb.reducedStorageForm;
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

  public int childrenLength () {
    return children == null ? 0 : children.length;
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

  public boolean compare (ClusterBasis cb) {
    if (cb.childrenLength() == 0 && childrenLength() == 0)
    { return cb.basis.minus(basis).normF() <= PsplHMatrixPack.epi; }
    else if (cb.childrenLength() > 0 && childrenLength() > 0 && cb.childrenLength() == childrenLength())
    {
      boolean equal = true;
      for (int i = 0; i < childrenLength(); i++)
      { equal &= children[i].compare(cb.children[i]); }
      return equal;
    }
    else
    { return cb.toMatrix().minus(toMatrix()).normF() <= PsplHMatrixPack.epi; }
  }

  public Matrix getTrans (int children_i) {
    if (reducedStorageForm && children_i >= 0 && children_i < childrenLength()) {
      int start_y = 0;
      for (int i = 0; i < children_i; i++)
      { start_y += children[i].getRank(); }
      int end_y = start_y + children[children_i].getRank() - 1;
      return basis.getMatrix(start_y, end_y, 0, basis.getColumnDimension() - 1);
    }
    else if (!reducedStorageForm)
    { System.out.println("Not in reduced storage form when retrieving Trans."); return null; }
    else
    { System.out.println("No children or invalid children index when retrieving Trans."); return null; }
  }

  public Matrix toMatrix() {
    if (!reducedStorageForm || children == null)
    { return basis; }
    else {
      int dim = 0; Matrix children_basis[] = new Matrix[children.length];
      for (int i = 0; i < children.length; i++) 
      { children_basis[i] = children[i].toMatrix(); dim += children_basis[i].getRowDimension(); }

      int start_x = 0, start_y = 0; 
      Matrix lower = new Matrix(dim, children.length * basis.getColumnDimension());
      for (int i = 0; i < children.length; i++)
      {
        int end_x = start_x + children_basis[i].getRowDimension() - 1, end_y = start_y + children_basis[i].getColumnDimension() - 1;
        lower.setMatrix(start_x, end_x, start_y, end_y, children_basis[i]);
        start_x = end_x + 1; start_y = end_y + 1;
      }
      
      return lower.times(basis);
    }
  }

  public Matrix toMatrix (int i) {
    return children[i].toMatrix();
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

      int start_x = 0, start_y = 0; 
      Matrix lower = new Matrix(dim, children.length * basis.getColumnDimension());
      for (int i = 0; i < children.length; i++)
      {
        int end_x = start_x + children_basis[i].getRowDimension() - 1, end_y = start_y + children_basis[i].getColumnDimension() - 1;
        lower.setMatrix(start_x, end_x, start_y, end_y, children_basis[i]);
        start_x = end_x + 1; start_y = end_y + 1;
      }

      Matrix temp = basis; basis = lower.transpose().times(temp);
      reducedStorageForm = true;
      return temp;
    }
  }


}