
import Jama.Matrix;

public class ClusterBasisProduct {

  private Matrix product;
  private ClusterBasisProduct[] children;

  public ClusterBasisProduct () {
    product = null;
    children = null;
  }

  public ClusterBasisProduct (Matrix product) {
    this.product = product;
    children = null;
  }

  public ClusterBasisProduct (int m) {
    product = null;
    children = new ClusterBasisProduct[m];
  }

  public ClusterBasisProduct (ClusterBasis left, ClusterBasis right) {
    int leftChild = left.childrenLength();
    int rightChild = right.childrenLength();
    if (leftChild == rightChild && leftChild > 0) {
      children = new ClusterBasisProduct[leftChild];
      for (int i = 0; i < leftChild; i++) {
        children[i] = new ClusterBasisProduct(left.getChildren()[i], right.getChildren()[i]);
      }
      product = collectProductSingle(left, right, left.getRank(), right.getRank());
    }
    else if (leftChild > 0 && rightChild == 0) {
      Matrix rightMat = right.toMatrix();
      children = new ClusterBasisProduct[leftChild];
      int y = 0;
      int x = rightMat.getColumnDimension() - 1;
      for (int i = 0; i < leftChild; i++) {
        ClusterBasis childI = left.getChildren()[i];
        int yEnd = y + childI.getDimension() - 1;
        children[i] = new ClusterBasisProduct(childI, rightMat.getMatrix(y, yEnd, 0, x));
        y = yEnd + 1;
      }
      product = collectProductSingle(left, null, left.getRank(), x + 1);    
    }
    else if (rightChild > 0 && leftChild == 0) {
      Matrix leftMat = left.toMatrix();
      children = new ClusterBasisProduct[rightChild];
      int y = leftMat.getRowDimension() - 1;
      int x = 0;
      for (int i = 0; i < rightChild; i++) {
        ClusterBasis childI = right.getChildren()[i];
        int xEnd = x + childI.getDimension() - 1;
        children[i] = new ClusterBasisProduct(leftMat.getMatrix(0, y, x, xEnd), childI);
        x = xEnd + 1;
      }
      product = collectProductSingle(null, right, y + 1, right.getRank());
    }
    else {
      children = null; 
      product = left.toMatrix().transpose().times(right.toMatrix()); 
    }
  }

  public ClusterBasisProduct (ClusterBasis left, Matrix right) {
    int leftChild = left.childrenLength();
    if (leftChild == 0) { 
      children = null; 
      product = left.toMatrix().transpose().times(right); 
    }
    else {
      children = new ClusterBasisProduct[leftChild];
      int y = 0;
      int x = right.getColumnDimension() - 1;
      for (int i = 0; i < leftChild; i++) {
        ClusterBasis childI = left.getChildren()[i];
        int yEnd = y + childI.getDimension() - 1;
        children[i] = new ClusterBasisProduct(childI, right.getMatrix(y, yEnd, 0, x));
        y = yEnd + 1;
      }
      product = collectProductSingle(left, null, left.getRank(), x + 1);
    }
  }

  public ClusterBasisProduct (Matrix left, ClusterBasis right) {
    int rightChild = right.childrenLength();
    if (rightChild == 0) { 
      children = null; 
      product = left.transpose().times(right.toMatrix()); 
    }
    else {
      children = new ClusterBasisProduct[rightChild];
      int y = left.getRowDimension() - 1;
      int x = 0;
      for (int i = 0; i < rightChild; i++) {
        ClusterBasis childI = right.getChildren()[i];
        int xEnd = x + childI.getDimension() - 1;
        children[i] = new ClusterBasisProduct(left.getMatrix(0, y, x, xEnd), childI);
        x = xEnd + 1;
      }
      product = collectProductSingle(null, right, y + 1, right.getRank());
    }
  }

  private Matrix collectProductSingle (ClusterBasis left, ClusterBasis right, int rankL, int rankR) {
    Matrix p = new Matrix (rankL, rankR);
    for (int i = 0; i < getNBlocks(); i++) {
      Matrix etI = left == null ? children[i].product : left.getTrans(i).transpose().times(children[i].product);
      Matrix eJ = right == null ? etI : etI.times(right.getTrans(i));
      p.plusEquals(eJ);
    }
    return p;
  }

  public int getNBlocks()
  { return children.length; }

  public Matrix getProduct () {
    return product;
  }

  public int childrenLength () {
    return children == null ? 0 : children.length;
  }

  public ClusterBasisProduct getChildren (int i) {
    return children == null ? null : children[i];
  }

  public ClusterBasisProduct[] setChildren (int m) {
    if (children == null) {
      children = new ClusterBasisProduct[m];
      for (int i = 0; i < m; i++)
      { children[i] = new ClusterBasisProduct(); }
    }
    return children;
  }

  public ClusterBasisProduct[] setChildren (ClusterBasisProduct[] children) {
    if (this.children == null) {
      this.children = children;
    }
    return this.children;
  }

  public Matrix getProduct (int i) {
    return getChildren(i) == null ? null : children[i].product;
  }

  public void accumProduct (Matrix product) {
    if (this.product == null)
    { this.product = new Matrix(product.getArrayCopy()); }
    else 
    { this.product.plusEquals(product); }
  }

  public void accumProduct (int i, Matrix product) {
    if (children[i] != null)
    { children[i].accumProduct(product); }
    else
    { children[i] = new ClusterBasisProduct(product); }
  }

  public void forwardTrans (ClusterBasis cb) {
    if (children != null && cb.childrenLength() > 0) {
      for (int i = 0; i < children.length; i++) {
        Matrix e = cb.getTrans(i);
        children[i].forwardTransRecur(cb.getChildren()[i], e);
      }
    }
  }

  private void forwardTransRecur (ClusterBasis cb, Matrix e) {
    product = product.times(e);
    if (children != null && cb.childrenLength() > 0) {
      for (int i = 0; i < children.length; i++) {
        Matrix ee = cb.getTrans(i).times(e);
        children[i].forwardTransRecur(cb.getChildren()[i], ee);
      }
    }
  }

  public Matrix accmAdmisBackward (ClusterBasis row, Matrix accm) {
    if (childrenLength() > 0 && row.childrenLength() > 0) {
      int y = 0;
      for (int i = 0; i < childrenLength(); i++) {
        Matrix p = getProduct();
        if (p != null) { 
          Matrix eI = row.getTrans(i);
          accumProduct(i, eI.times(p));
        }
        int yEnd = y + row.getChildren()[i].getDimension() - 1;
        Matrix accmSub = accm.getMatrix(y, yEnd, 0, accm.getColumnDimension() - 1);
        accmSub = children[i].accmAdmisBackward(row.getChildren()[i], accmSub);
        if (accmSub != null)
        { accm.setMatrix(y, yEnd, 0, accm.getColumnDimension() - 1, accmSub); }
        y = yEnd + 1;
      }
      return accm;
    }
    else if (getProduct() != null) {
      accm.plusEquals(row.toMatrix().times(getProduct()));
      return accm;
    }
    else {
      return null;
    }
  }


  public static Matrix alignRank (Matrix s, int row, int col) {
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

  public static Matrix basisInteract (Matrix vec, Block b, ClusterBasisProduct forward, ClusterBasisProduct accmAdmis, ClusterBasis row, boolean transpose) {
    if (b.castH2Matrix() != null) {
      H2Matrix h2 = b.castH2Matrix();
      Matrix[] vecP = transpose ? h2.getRowBasis().partitionMatrix(vec) : h2.getColBasis().partitionMatrix(vec);
      return basisInteract(vecP, h2, forward, accmAdmis, row, transpose);
    }
    else if (b.getType() == Block.Block_t.DENSE) {
      Matrix d = transpose ? b.toDense().transpose() : b.toDense();
      return d.times(vec);
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

  private static Matrix basisInteract (Matrix[] vec, H2Matrix h2, ClusterBasisProduct forward, ClusterBasisProduct accmAdmis, ClusterBasis row, boolean transpose) {
    int m = transpose ? h2.getNColumnBlocks() : h2.getNRowBlocks();
    int n = transpose ? h2.getNRowBlocks() : h2.getNColumnBlocks();
    int rank = vec[0].getColumnDimension();
    int[] dims = new int[m];
    int dim = 0;

    Matrix[] accm = new Matrix[m];
    boolean skipDense = true;
    ClusterBasisProduct[] accmChildren = accmAdmis.setChildren(m);


    for (int i = 0; i < m; i++) {
      dims[i] = transpose ? h2.getColumnDimension(i) : h2.getRowDimension(i);
      dim += dims[i];
      Matrix accmI = new Matrix(dims[i], rank);
      for (int j = 0; j < n; j++) {
        Block eIJ = transpose ? h2.getElement(j, i) : h2.getElement(i, j);
        Matrix accmIJ = basisInteract(vec[j], eIJ, forward.getChildren(j), accmChildren[i], row.getChildren()[i], transpose);
        if (accmIJ != null) {
          skipDense = false;
          accmI.plusEquals(accmIJ);
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


}