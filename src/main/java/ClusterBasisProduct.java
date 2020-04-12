
import Jama.Matrix;

public class ClusterBasisProduct {

  private Matrix product;
  private ClusterBasisProduct children[];

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

  public int getNBlocks()
  { return children.length; }

  public Matrix getProduct () {
    return product;
  }

  public ClusterBasisProduct getChildren (int i) {
    return children == null ? null : children[i];
  }

  public Matrix getProduct (int i) {
    return getChildren(i) == null ? null : children[i].product;
  }

  public void accumProduct (Matrix product) {
    if (this.product == null)
    { this.product = new Matrix(product.getArray()); }
    else 
    { this.product.plusEquals(product); }
  }

  public void accumProduct (int i, Matrix product) {
    if (children[i] != null)
    { children[i].accumProduct(product); }
    else
    { children[i] = new ClusterBasisProduct(product); }
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



}