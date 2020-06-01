
import Jama.Matrix;
import Jama.SingularValueDecomposition;

public class ClusterBasis {

  private Matrix basis;
  private ClusterBasis[] children;
  private int xyStart;
  private int[] childDim;
  private ClusterBasis parent;

  public ClusterBasis (int m, int n) {
    basis = new Matrix(m, n);
    children = null;
    xyStart = 0;
    childDim = null;
    parent = null;
  }

  public ClusterBasis (Matrix m) {
    basis = new Matrix(m.getArrayCopy());
    children = null;
    xyStart = 0;
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

    this.xyStart = xyStart;
    childDim = null;
    parent = null;
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
    return xyStart;
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

  private Matrix[] partitionMatrix (Matrix m) {
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
    Matrix l = new Matrix (childDim[children.length], rank);
    for (int i = 0; i < children.length; i++) 
    { l.setMatrix(childDim[i], childDim[i + 1] - 1, 0, rank - 1, projChild[i].getProduct()); }

    ClusterBasisProduct proj = updateAdditionalBasisLeaf(l);
    proj.setChildren(projChild);
    return proj;
  }

  private ClusterBasisProduct updateAdditionalBasisLeaf (Matrix m) {
    Matrix v = basis.times(basis.transpose());

    int size = basis.getRowDimension();
    Matrix f = m.times(m.transpose());

    Matrix projLeft = Matrix.identity(size, size).minus(v);
    Matrix projRight = projLeft.transpose();

    Matrix g = projLeft.times(f).times(projRight);

    if (g.normF() / size / size <= PsplHMatrixPack.EPI)
    { return new ClusterBasisProduct(getBasis().transpose().times(m)); }

    SingularValueDecomposition svdd = g.svd();
    double[] s = svdd.getSingularValues(); 
    int rank = 0;
    while (rank < s.length && s[rank] > PsplHMatrixPack.EPI)
    { rank++; }

    Matrix newBasis = appendAdditionalBasis(svdd.getU().getMatrix(0, size - 1, 0, rank));
    return new ClusterBasisProduct(newBasis.transpose().times(m));
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
    Matrix accmY = h2matrixTimesInteract(h2, forward, accmAdmis, row, transpose);
    
    if (accmY == null)
    { accmY = new Matrix(row.getDimension(), getRank()); }
    accmY = h2matrixTimesBackward(accmAdmis, row, accmY);
    return accmY;
  }

  private Matrix h2matrixTimesInteract (H2Matrix h2, ClusterBasisProduct forward, ClusterBasisProduct accmAdmis, ClusterBasis row, boolean transpose) {
    int m = transpose ? h2.getNColumnBlocks() : h2.getNRowBlocks();
    int n = transpose ? h2.getNRowBlocks() : h2.getNColumnBlocks();
    int rank = getRank();
    Matrix[] accm = new Matrix[m];
    boolean skipDense = true;
    ClusterBasisProduct[] accmChildren = accmAdmis.setChildren(m);
    int[] dims = new int[m];
    int dim = 0;

    for (int i = 0; i < m; i++) {
      dims[i] = transpose ? h2.getColumnDimension(i) : h2.getRowDimension(i);
      dim += dims[i];
      Matrix accmI = new Matrix(dims[i], rank);
      for (int j = 0; j < n; j++) {
        Block eIJ = transpose ? h2.getElement(j, i) : h2.getElement(i, j);
        Matrix accmIJ = children[j].h2matrixTimesInteract(eIJ, forward.getChildren(j), accmChildren[i], row.children[i], transpose);
        if (accmIJ != null) {
          skipDense = false;
          Matrix accmE = accmIJ.times(getTrans(j));
          accmI.plusEquals(accmE);
        }
      }
      accm[i] = accmI;
    }

    if (skipDense)
    { return null; }

    int y = 0;
    Matrix accmY = new Matrix(dim, rank);
    for (int i = 0; i < m; i++) {
      int yEnd = y + dims[i] - 1;
      accmY.setMatrix(y, yEnd, 0, accm[i].getColumnDimension() - 1, accm[i]);
      y = yEnd + 1;
    }
    return accmY;
  }

  private Matrix alignRank (Matrix s, int row, int col) {
    int rowS = s.getRowDimension();
    int colS = s.getColumnDimension();
    if (rowS > row && colS > col) 
    { return s.getMatrix(0, row - 1, 0, col - 1); }
    else if (rowS < row || colS < col) {
      Matrix r = new Matrix(row, col);
      rowS = Integer.min(rowS, row);
      colS = Integer.min(colS, col);
      r.setMatrix(0, rowS - 1, 0, colS - 1, s); 
      return r; 
    }
    else
    { return s; }
  }

  private Matrix h2matrixTimesInteract (Block b, ClusterBasisProduct forward, ClusterBasisProduct accmAdmis, ClusterBasis row, boolean transpose) {
    if (b.castH2Matrix() != null) {
      H2Matrix h2 = b.castH2Matrix();
      return h2matrixTimesInteract(h2, forward, accmAdmis, row, transpose);
    }
    else if (b.getType() == Block.Block_t.DENSE) {
      Matrix d = transpose ? b.toDense().transpose() : b.toDense();
      return d.times(toMatrix());
    }
    else if (b.getType() == Block.Block_t.LOW_RANK) {
      LowRank lr = b.toLowRank();
      Matrix s = transpose ? lr.getS().transpose() : lr.getS();
      Matrix r = alignRank(s, row.getRank(), forward.getProduct().getRowDimension());
      Matrix m = r.times(forward.getProduct());
      accmAdmis.accumProduct(m);
      return null;
    }
    else {
      return null;
    }
  }

  private Matrix h2matrixTimesBackward (ClusterBasisProduct accmAdmis, ClusterBasis row, Matrix accm) {
    if (accmAdmis.childrenLength() > 0 && row.childrenLength() > 0) {
      int y = 0;
      for (int i = 0; i < children.length; i++) {
        Matrix p = accmAdmis.getProduct();
        if (p != null) { 
          Matrix eI = row.getTrans(i);
          accmAdmis.accumProduct(i, eI.times(p));
        }
        int yEnd = y + row.getChildren()[i].getDimension() - 1;
        Matrix accmSub = accm.getMatrix(y, yEnd, 0, accm.getColumnDimension() - 1);
        accmSub = h2matrixTimesBackward(accmAdmis.getChildren(i), row.getChildren()[i], accmSub);
        if (accmSub != null)
        { accm.setMatrix(y, yEnd, 0, accm.getColumnDimension() - 1, accmSub); }
        y = yEnd + 1;
      }
      return accm;
    }
    else if (accmAdmis.getProduct() != null) {
      accm.plusEquals(row.toMatrix().times(accmAdmis.getProduct()));
      return accm;
    }
    else {
      return null;
    }
  }


}