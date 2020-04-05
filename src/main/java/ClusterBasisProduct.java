
import Jama.Matrix;

public class ClusterBasisProduct {

  private Matrix product;
  private ClusterBasisProduct children[][];

  public ClusterBasisProduct (Matrix product) {
    this.product = product;
    children = null;
  }

  public ClusterBasisProduct (int m, int n) {
    product = null;
    children = new ClusterBasisProduct[m][n];
  }

  public ClusterBasisProduct (ClusterBasis left, ClusterBasis right) {
    int left_child = left.childrenLength(), right_child = right.childrenLength();

    if (left_child == 0 || right_child == 0) 
    { children = null; product = left.toMatrix().transpose().times(right.toMatrix()); }
    else {
      children = new ClusterBasisProduct[left_child][right_child];

      for (int i = 0; i < left_child; i++) {
        for (int j = 0; j < right_child; j++) {
          children[i][j] = new ClusterBasisProduct(left.getChildren()[i], right.getChildren()[j]);
        }
      }

      product = collectProduct(left, right);
    }
  }

  public ClusterBasisProduct (ClusterBasis left_prime, ClusterBasis right_prime, ClusterBasisProduct X, ClusterBasisProduct Y, H2Matrix h) {
    if (left_prime.childrenLength() == 0 || right_prime.childrenLength() == 0) {
      product = left_prime.toMatrix().transpose().times(h.toDense()).times(right_prime.toMatrix());
      children = null;
    }
    else {
      int m = h.getNRowBlocks(), n = h.getNColumnBlocks();
      children = new ClusterBasisProduct[m][n];

      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          Block b = h.getElement(i, j);
          if (b.getType() == Block.Block_t.LOW_RANK) {
            Matrix product = X.getProduct(i, i).times(b.toLowRank().getS()).times(Y.getProduct(j, j));
            children[i][j] = new ClusterBasisProduct(product);
          } 
          else if (b.getType() == Block.Block_t.DENSE) {
            Matrix product = left_prime.toMatrix(i).transpose().times(b.toDense()).times(right_prime.toMatrix(j));
            children[i][j] = new ClusterBasisProduct(product);
          }
          else {
            children[i][j] = new ClusterBasisProduct(left_prime.getChildren()[i], right_prime.getChildren()[j], X.getChildren(i, i), Y.getChildren(j, j), b.castH2Matrix());
          }
        }
      }
      product = collectProduct(left_prime, right_prime);
    }
  }

  public ClusterBasisProduct (ClusterBasis left, ClusterBasis right, Matrix product_upper) {
    product = product_upper;
    int m = left.childrenLength(), n = right.childrenLength();
    children = new ClusterBasisProduct[m][n];

    for (int i = 0; i < m; i++) {
      Matrix E_i = left.getTrans(i).times(product_upper);
      for (int j = 0; j < n; j++) {
        Matrix Et_j = right.getTrans(j).transpose();
        children[i][j] = new ClusterBasisProduct(E_i.times(Et_j));
      }
    }
  }

  public int getNRowBlocks()
  { return children.length; }

  public int getNColumnBlocks()
  { return children[0].length; }

  public Matrix getProduct () {
    return product;
  }

  public Matrix getProduct (int i, int j) {
    return getChildren(i, j) == null ? null : children[i][j].product;
  }

  public void accumProduct (int i, int j, Matrix product) {
    if (this.product == null)
    { this.product = new Matrix(product.getArray()); }
    else 
    { this.product.plusEquals(product); }
  }

  public void expandChildren (int i, int j, int m, int n) {
    if (children[i][j] == null)
    { children[i][j] = new ClusterBasisProduct(m, n); }
  }

  public Matrix collectProduct (ClusterBasis left, ClusterBasis right) {
    Matrix product = new Matrix (left.getRank(), right.getRank());

    for (int i = 0; i < getNRowBlocks(); i++) {
      Matrix Et_i = left.getTrans(i).transpose();
      for (int j = 0; j < getNColumnBlocks(); j++) {
        Matrix E_j = right.getTrans(j);
        product.plusEquals(Et_i.times(children[i][j].product).times(E_j));
      }
    }

    return product;
  }

  public ClusterBasisProduct getChildren (int i, int j) {
    return children == null ? null : children[i][j];
  }


}