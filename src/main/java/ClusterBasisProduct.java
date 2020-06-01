
import Jama.Matrix;

public class ClusterBasisProduct {

  private Matrix product;
  private ClusterBasisProduct children[];

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
    int left_child = left.childrenLength(), right_child = right.childrenLength();
    if (left_child == 0 || right_child == 0 || left_child != right_child) 
    { children = null; product = left.toMatrix().transpose().times(right.toMatrix()); }
    else {
      children = new ClusterBasisProduct[left_child];
      for (int i = 0; i < left_child; i++) {
        children[i] = new ClusterBasisProduct(left.getChildren()[i], right.getChildren()[i]);
      }
      product = collectProduct_single(left, right);
    }
  }

  public Matrix collectProduct_single (ClusterBasis left, ClusterBasis right) {
    Matrix product = new Matrix (left.getRank(), right.getRank());
    for (int i = 0; i < getNBlocks(); i++) {
      Matrix Et_i = left.getTrans(i).transpose();
      Matrix E_j = right.getTrans(i);
      product.plusEquals(Et_i.times(children[i].product).times(E_j));
    }
    return product;
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


}