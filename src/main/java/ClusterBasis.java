
import java.util.Arrays;

import Jama.Matrix;
import Jama.QRDecomposition;
import Jama.SingularValueDecomposition;

public class ClusterBasis {

  private Matrix basis;
  private ClusterBasis[] children;
  private int[] childDim;
  private ClusterBasis parent;

  public ClusterBasis (int m, int n) {
    basis = new Matrix(m, n);
    children = null;
    childDim = null;
    parent = null;
  }

  public ClusterBasis (Matrix m) {
    basis = new Matrix(m.getArrayCopy());
    children = null;
    childDim = null;
    parent = null;
  }

  public ClusterBasis (int xyStart, int mn, boolean rowB, int nleaf, int partStrat, int rank, double admis, PsplHMatrixPack.DataFunction func) {

    if (rowB)
    { basis = Dense.getBasisU(xyStart, mn, rank, admis, func); }
    else
    { basis = Dense.getBasisVT(xyStart, mn, rank, admis, func); }

    if (mn > nleaf) {
      int mnBlock = mn / partStrat;
      int mnRemain = mn - (partStrat - 1) * mnBlock;
      children = new ClusterBasis[partStrat];
  
      for (int i = 0; i < partStrat; i++) {
        int mnE = i == partStrat - 1 ? mnRemain : mnBlock;
        int xyE = xyStart + mnBlock * i;
        children[i] = new ClusterBasis (xyE, mnE, rowB, nleaf, partStrat, rank, admis, func);
        children[i].parent = this;
      }
    }
    else
    { children = null; }

    childDim = null;
    parent = null;
  }

  public ClusterBasis (Hierarchical h, boolean rowB, int rank) {
    children = null;
    childDim = null;
    parent = null;
    if (rowB) {
      basis = new Matrix(h.getRowDimension(), rank);
      spanChildrenRow(h, rank); 
    }
    else {
      basis = new Matrix(h.getColumnDimension(), rank); 
      spanChildrenCol(h, rank); 
    }
    orthogonalize();
  }

  private void spanChildrenRow (Hierarchical h, int rank) {

    int m = h.getNRowBlocks();
    int n = h.getNColumnBlocks();

    if (children == null) { 
      children = new ClusterBasis[m];
      for (int i = 0; i < m; i++) {
        int dim = h.getElement(i, 0).getRowDimension();
        children[i] = new ClusterBasis(dim, rank); 
        children[i].parent = this; 
      }
    }

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        Block b = h.getElement(i, j);

        if (b.getType() == Block.Block_t.LOW_RANK)
        { children[i].accBasis(b.toDense(), rank); }
        else if (b.getType() == Block.Block_t.HIERARCHICAL)
        { children[i].spanChildrenRow(b.castHierarchical(), rank); }
      }
    }

    propagateBasis();
  }

  private void spanChildrenCol (Hierarchical h, int rank) {

    int m = h.getNRowBlocks();
    int n = h.getNColumnBlocks();

    if (children == null) { 
      children = new ClusterBasis[n];
      for (int i = 0; i < n; i++) {
        int dim = h.getElement(0, i).getColumnDimension();
        children[i] = new ClusterBasis(dim, rank); 
        children[i].parent = this; 
      }
    }

    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        Block b = h.getElement(i, j);
        if (b.getType() == Block.Block_t.LOW_RANK)
        { children[j].accBasis(b.toDense().transpose(), rank); }
        else if (b.getType() == Block.Block_t.HIERARCHICAL)
        { children[j].spanChildrenCol(b.castHierarchical(), rank); }
      }
    }

    propagateBasis();
  }

  private void propagateBasis () {
    if (basis != null) {
      Matrix[] mPart = partitionMatrix(basis);
      for (int i = 0; i < children.length; i++)
      { children[i].accBasis(mPart[i]); }
    }
  }

  private void orthogonalize () {
    if (basis != null) {
      int dim = basis.getRowDimension();
      int rank = basis.getColumnDimension();
      QRDecomposition qrd = basis.qr();
      basis = qrd.getQ().getMatrix(0, dim - 1, 0, rank - 1);
    }
    if (children != null) {
      for (int i = 0; i < childrenLength(); i++)
      { children[i].orthogonalize(); }
    }
  }

  public ClusterBasis copyClusterBasis () {
    ClusterBasis c = new ClusterBasis(basis);
    int l = childrenLength();
    if (l > 0) {
      c.children = new ClusterBasis[l];
      for (int i = 0; i < l; i++) { 
        c.children[i] = children[i].copyClusterBasis(); 
        c.children[i].parent = c; 
      }
    }
    if (childDim != null) {
      c.childDim = Arrays.copyOf(childDim, l + 1);
    }
    return c;
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

  public int childrenLength () {
    return children == null ? 0 : children.length;
  }

  public int getPartStrat () {
    return children.length;
  }

  public void partition (int m) {
    if (children == null && m > 0) {
      children = new ClusterBasis[m];
      int y = basis.getRowDimension();
      int yStep = y / m;
      int yStart = 0;
      for (int i = 0; i < m; i++) {
        int yEnd = Integer.min(yStart + yStep, y) - 1;
        children[i] = new ClusterBasis(basis.getMatrix(yStart, yEnd, 0, basis.getColumnDimension() - 1));
        yStart = yEnd + 1;
      }
    }
  }

  public Matrix getBasis () {
    return basis;
  }

  public Matrix accBasis (Matrix m, int rank) {
    Matrix r = Matrix.random(m.getColumnDimension(), rank);
    Matrix mr = m.times(r);
    if (basis == null)
    { basis = mr.copy(); }
    else
    { basis.plusEquals(mr); }
    return basis;
  }

  public Matrix accBasis (Matrix m) {
    if (basis == null)
    { basis = m.copy(); }
    else
    { basis.plusEquals(m); }
    return basis;
  }

  public ClusterBasis[] getChildren () {
    return children;
  }

  public boolean compare (ClusterBasis cb) {
    if (this == cb)
    { return true; }
    else if (cb.childrenLength() == 0 && childrenLength() == 0) {
      double norm = cb.basis.minus(basis).normF() / basis.getRowDimension() / basis.getColumnDimension();
      return norm <= PsplHMatrixPack.EPI; 
    }
    else if (cb.childrenLength() > 0 && childrenLength() > 0 && cb.childrenLength() == childrenLength()) {
      boolean equal = true;
      for (int i = 0; i < childrenLength(); i++)
      { equal &= children[i].compare(cb.children[i]); }
      return equal;
    }
    else {
      int rank = Integer.min(cb.getRank(), getRank());
      return cb.toMatrix(rank).minus(toMatrix(rank)).normF() / getDimension() / rank <= PsplHMatrixPack.EPI; 
    }
  }

  public Matrix getTrans (int childrenI) {
    if (childDim != null && childrenI >= 0 && childrenI < childrenLength()) {
      int startY = childDim[childrenI];
      int endY = childDim[childrenI + 1] - 1;
      return basis.getMatrix(startY, endY, 0, basis.getColumnDimension() - 1);
    }
    else if (childDim == null)
    { PsplHMatrixPack.errorOut("Not in reduced storage form when retrieving Trans."); return null; }
    else
    { PsplHMatrixPack.errorOut("No children or invalid children index when retrieving Trans."); return null; }
  }

  public Matrix toMatrix() {
    if (childDim == null || children == null)
    { return getBasis(); }
    else {
      int dim = 0;
      Matrix[] childrenBasis = new Matrix[children.length];
      for (int i = 0; i < children.length; i++) 
      { childrenBasis[i] = children[i].toMatrix(); dim += childrenBasis[i].getRowDimension(); }

      int startX = 0;
      int startY = 0; 
      Matrix lower = new Matrix(dim, childDim[children.length]);
      for (int i = 0; i < children.length; i++) {
        int endX = startX + childrenBasis[i].getRowDimension() - 1;
        int endY = startY + childrenBasis[i].getColumnDimension() - 1;
        lower.setMatrix(startX, endX, startY, endY, childrenBasis[i]);
        startX = endX + 1; startY = endY + 1;
      }
      
      return lower.times(basis);
    }
  }

  public Matrix toMatrix (int maxRank) {
    Matrix m = toMatrix();
    if (m.getColumnDimension() > maxRank)
    { m = m.getMatrix(0, m.getRowDimension() - 1, 0, maxRank - 1); }
    return m;
  }

  public Matrix childMatrix (int i) {
    return children[i].toMatrix();
  }

  public Matrix convertReducedStorageForm() {
    if (children == null)
    { return basis; }
    else if (childDim != null)
    { return toMatrix(); }
    else {
      int dim = 0; 
      Matrix[] childrenBasis = new Matrix[children.length];
      childDim = new int[children.length + 1];
      childDim[0] = 0;
      for (int i = 0; i < children.length; i++) { 
        childrenBasis[i] = children[i].convertReducedStorageForm();
        childDim[i + 1] = childDim[i] + childrenBasis[i].getColumnDimension();
        dim += childrenBasis[i].getRowDimension();
      }

      int startX = 0;
      int startY = 0; 
      Matrix lower = new Matrix(dim, children.length * basis.getColumnDimension());
      for (int i = 0; i < children.length; i++) {
        int endX = startX + childrenBasis[i].getRowDimension() - 1;
        int endY = startY + childrenBasis[i].getColumnDimension() - 1;
        lower.setMatrix(startX, endX, startY, endY, childrenBasis[i]);
        startX = endX + 1; startY = endY + 1;
      }

      Matrix temp = basis; basis = lower.transpose().times(temp);
      return temp;
    }
  }

  public void updateTrans () {
    if (children != null && childDim != null) {
      int[] newChildDim = new int[children.length + 1];
      newChildDim[0] = 0;
      for (int i = 0; i < children.length; i++) 
      { newChildDim[i + 1] = newChildDim[i] + children[i].getRank(); }

      if (newChildDim[children.length] != childDim[children.length]) {
        Matrix newBasis = new Matrix(newChildDim[children.length], basis.getColumnDimension());
        for (int i = 0; i < children.length; i++) {
          Matrix trans = getTrans(i);
          int startY = newChildDim[i];
          int endY = Integer.min(startY + trans.getRowDimension() - 1, newChildDim[i + 1] - 1);
          newBasis.setMatrix(startY, endY, 0, basis.getColumnDimension() - 1, trans); 
        }
        basis = newBasis;
        childDim = newChildDim;
      }
    }
  }

  public ClusterBasisProduct updateAdditionalBasis (Matrix m) {
    Matrix mA = m.getRowDimension() == getDimension() ? m : m.transpose();
    ClusterBasisProduct product;
    if (childDim != null && children != null) 
    { product = updateAdditionalBasisNonLeaf(mA); }
    else 
    { product = updateAdditionalBasisLeaf(mA); }
    if (parent != null)
    { parent.updateTrans(); }
    return product;
  }

  public Matrix[] partitionMatrix (Matrix m) {
    Matrix[] mPart = new Matrix[children.length];
    int startY = 0;
    for (int i = 0; i < children.length; i++) {
      int endY = startY + children[i].getDimension() - 1;
      mPart[i] = m.getMatrix(startY, endY, 0, m.getColumnDimension() - 1);
      startY = endY + 1;
    }
    return mPart;
  }

  private ClusterBasisProduct updateAdditionalBasisRecur (Matrix m) {
    if (childDim != null && children != null) 
    { return updateAdditionalBasisNonLeaf(m); }
    else 
    { return updateAdditionalBasisLeaf(m); }
  }

  private ClusterBasisProduct updateAdditionalBasisNonLeaf (Matrix m) {
    ClusterBasisProduct[] projChild = new ClusterBasisProduct[children.length];
    Matrix[] mPart = partitionMatrix(m);
    for (int i = 0; i < children.length; i++)
    { projChild[i] = children[i].updateAdditionalBasisRecur(mPart[i]); }
    updateTrans();

    int rank = m.getColumnDimension();
    int size = childDim[children.length];
    Matrix l = new Matrix (size, rank);
    for (int i = 0; i < children.length; i++) 
    { l.setMatrix(childDim[i], childDim[i + 1] - 1, 0, rank - 1, projChild[i].getProduct()); }

    QRDecomposition qrb = basis.qr();
    Matrix basisQ = qrb.getQ();
    Matrix g = projectExcludeBasis(l, basisQ);

    ClusterBasisProduct proj;
    if (g.normF() <= PsplHMatrixPack.EPI * size * size)
    { proj = new ClusterBasisProduct(qrb.solve(l)); }
    else {
      SingularValueDecomposition svdd = g.svd();
      double[] s = svdd.getSingularValues(); 
      int rankAdd = 0;
      while (rankAdd < s.length && s[rankAdd] > PsplHMatrixPack.EPI)
      { rankAdd++; }

      Matrix newBasis = appendAdditionalBasis(svdd.getU().getMatrix(0, size - 1, 0, rankAdd));
      proj = new ClusterBasisProduct(newBasis.solve(l));
    }

    proj.setChildren(projChild);
    return proj;
  }

  private ClusterBasisProduct updateAdditionalBasisLeaf (Matrix m) {
    Matrix g = projectExcludeBasis(m, basis);
    int size = basis.getRowDimension();

    if (g.normF() <= PsplHMatrixPack.EPI * size * size)
    { return new ClusterBasisProduct(getBasis().transpose().times(m)); }

    SingularValueDecomposition svdd = g.svd();
    double[] s = svdd.getSingularValues(); 
    int rank = 0;
    while (rank < s.length && s[rank] > PsplHMatrixPack.EPI)
    { rank++; }

    Matrix newBasis = appendAdditionalBasis(svdd.getU().getMatrix(0, size - 1, 0, rank));
    return new ClusterBasisProduct(newBasis.transpose().times(m));
  }

  private Matrix projectExcludeBasis (Matrix m, Matrix basisQ) {
    Matrix v = basisQ.times(basisQ.transpose());
    int size = basisQ.getRowDimension();
    Matrix f = m.times(m.transpose());

    Matrix projLeft = Matrix.identity(size, size).minus(v);
    Matrix projRight = projLeft.transpose();

    return projLeft.times(f).times(projRight);
  } 

  public Matrix appendAdditionalBasis (Matrix add) {
    if (basis == null) { 
      basis = add.copy(); 
    }
    else {
      Matrix newBasis = new Matrix(basis.getRowDimension(), basis.getColumnDimension() + add.getColumnDimension());
      newBasis.setMatrix(0, basis.getRowDimension() - 1, 0, basis.getColumnDimension() - 1, basis);
      newBasis.setMatrix(0, basis.getRowDimension() - 1, basis.getColumnDimension(), basis.getColumnDimension() + add.getColumnDimension() - 1, add);
      basis = newBasis;
    }
    return getBasis();
  }

  public Matrix h2matrixTimes (H2Matrix h2, boolean transpose) {
    ClusterBasis col = transpose ? h2.getRowBasis() : h2.getColBasis();
    ClusterBasisProduct forward = new ClusterBasisProduct(col, this);
    forward.forwardTrans(this);

    ClusterBasis row = transpose ? h2.getColBasis() : h2.getRowBasis();
    ClusterBasisProduct accmAdmis = new ClusterBasisProduct();
    Matrix accmY = ClusterBasisProduct.basisInteract(toMatrix(), h2, forward, accmAdmis, row, transpose);
    
    if (accmY == null)
    { accmY = new Matrix(row.getDimension(), getRank()); }
    accmY = accmAdmis.accmAdmisBackward(row, accmY);
    return accmY;
  }


}