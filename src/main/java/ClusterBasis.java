
import Jama.Matrix;
import Jama.SingularValueDecomposition;

public class ClusterBasis {

  private Matrix basis;
  private Matrix basis_add;
  private ClusterBasis children[];
  private int xy_start;
  private boolean reducedStorageForm;

  public ClusterBasis (int m, int n) {
    basis = new Matrix(m, n);
    basis_add = null;
    children = null;
    xy_start = 0;
    reducedStorageForm = false;
  }

  public ClusterBasis (Matrix m) {
    basis = new Matrix(m.getArrayCopy());
    basis_add = null;
    children = null;
    xy_start = 0;
    reducedStorageForm = false;
  }

  public ClusterBasis (ClusterBasis cb, Matrix m) {
    basis = cb.basis.times(m);
    basis_add = null;
    children = cb.children;
    xy_start = cb.xy_start;
    reducedStorageForm = cb.reducedStorageForm;
  }

  public ClusterBasis (int xy_start, int mn, boolean row_col, int nleaf, int part_strat, int rank, double admis, PsplHMatrixPack.dataFunction func) {

    if (row_col)
    { basis = Dense.getBasisU(xy_start, mn, rank, admis, func); }
    else
    { basis = Dense.getBasisVT(xy_start, mn, rank, admis, func); }
    basis_add = null;

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
    return basis.getColumnDimension() + (basis_add == null ? 0 : basis_add.getColumnDimension());
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

  public Matrix getBasis () {
    if (basis_add == null)
    { return basis; }
    else {
      Matrix new_basis = new Matrix(basis.getRowDimension(), basis_add.getColumnDimension() + basis.getColumnDimension());
      new_basis.setMatrix(0, basis.getRowDimension() - 1, 0, basis.getColumnDimension() - 1, basis);
      new_basis.setMatrix(0, basis.getRowDimension() - 1, basis.getColumnDimension(), basis.getColumnDimension() + basis_add.getColumnDimension() - 1, basis_add);
      return new_basis;
    }
  }

  public ClusterBasis[] getChildren () {
    return children;
  }

  public boolean compare (ClusterBasis cb) {
    if (this == cb)
    { return true; }
    else if (cb.childrenLength() == 0 && childrenLength() == 0)
    { return cb.getBasis().minus(getBasis()).normF() <= PsplHMatrixPack.epi; }
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
    { return getBasis(); }
    else {
      int dim = 0; Matrix children_basis[] = new Matrix[children.length];
      for (int i = 0; i < children.length; i++) 
      { children_basis[i] = children[i].toMatrix(); dim += children_basis[i].getRowDimension(); }

      int start_x = 0, start_y = 0; 
      Matrix lower = new Matrix(dim, children.length * basis.getColumnDimension());
      for (int i = 0; i < children.length; i++) {
        int end_x = start_x + children_basis[i].getRowDimension() - 1, end_y = start_y + children_basis[i].getColumnDimension() - 1;
        lower.setMatrix(start_x, end_x, start_y, end_y, children_basis[i]);
        start_x = end_x + 1; start_y = end_y + 1;
      }
      
      return lower.times(basis);
    }
  }

  public Matrix toMatrix (int max_rank) {
    Matrix m = toMatrix();
    if (m.getColumnDimension() > max_rank)
    { m = m.getMatrix(0, m.getRowDimension() - 1, 0, max_rank - 1); }
    return m;
  }

  public Matrix childMatrix (int i) {
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
      for (int i = 0; i < children.length; i++) {
        int end_x = start_x + children_basis[i].getRowDimension() - 1, end_y = start_y + children_basis[i].getColumnDimension() - 1;
        lower.setMatrix(start_x, end_x, start_y, end_y, children_basis[i]);
        start_x = end_x + 1; start_y = end_y + 1;
      }

      Matrix temp = basis; basis = lower.transpose().times(temp);
      reducedStorageForm = true;
      return temp;
    }
  }

  public Matrix appendAdditionalBasis (Matrix add) {
    if (basis_add == null)
    { basis_add = new Matrix(add.getArrayCopy()); }
    else {
      Matrix new_basis = new Matrix(basis_add.getRowDimension(), basis_add.getColumnDimension() + add.getColumnDimension());
      new_basis.setMatrix(0, basis_add.getRowDimension() - 1, 0, basis_add.getColumnDimension() - 1, basis_add);
      new_basis.setMatrix(0, basis_add.getRowDimension() - 1, basis_add.getColumnDimension(), basis_add.getColumnDimension() + add.getColumnDimension() - 1, add);
      basis_add = new_basis;
    }
    return getBasis();
  }

  public Matrix updateAdditionalBasis (Matrix m) {
    Matrix V = basis.times(basis.transpose());
    if (basis_add != null)
    { V.plusEquals(basis_add.times(basis_add.transpose())); }

    int size = basis.getRowDimension();
    boolean row_col = m.getRowDimension() == size;
    if (!row_col && m.getColumnDimension() != size)
    { System.out.println("Expected F size: " + size); return null; }

    Matrix F = row_col ? m.times(m.transpose()) : m.transpose().times(m);

    Matrix proj_left = Matrix.identity(size, size).minus(V);
    Matrix proj_right = proj_left.transpose();

    Matrix G = proj_left.times(F).times(proj_right);

    SingularValueDecomposition svd_ = G.svd();
    double[] s = svd_.getSingularValues(); int rank = 0;
    while (rank < s.length && s[rank] > PsplHMatrixPack.epi)
    { rank++; }
    return appendAdditionalBasis(svd_.getU().getMatrix(0, size - 1, 0, rank));
  }


}