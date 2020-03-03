
import Jama.Matrix;

public class ClusterBasisU {

  private Matrix basis;
  private ClusterBasisU children[];
  private int y_start;
  private boolean reducedStorageForm;

  public ClusterBasisU (int y_start, int m, int nleaf, int part_strat, int rank, double admis, PsplHMatrixPack.dataFunction func) {

    basis = Dense.getBasisU(y_start, m, rank, admis, func);

    if (m > nleaf) {
      int m_block = m / part_strat, m_remain = m - (part_strat - 1) * m_block;
      children = new ClusterBasisU[part_strat];
  
      for (int i = 0; i < part_strat; i++) {
        int m_e = i == part_strat - 1 ? m_remain : m_block;
        int y_e = y_start + m_block * i;
        children[i] = new ClusterBasisU (y_e, m_e, nleaf, part_strat, rank, admis, func);
      }
    }
    else
    { children = null; }

    this.y_start = y_start;
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
    if (children != null)
    { return children[0].getRank(); }
    else
    { return basis.getColumnDimension(); }
  }

  public boolean hasChildren () {
    return children != null;
  }

  public ClusterBasisU[] getChildren() {
    return children;
  }

  public Matrix toMatrix() {
    System.out.println("x");
    return basis;
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
      Matrix temp = basis; basis = temp.transpose().times(lower);
      return temp;
    }
  }


}