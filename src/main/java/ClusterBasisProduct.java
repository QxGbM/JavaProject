
import Jama.Matrix;

public class ClusterBasisProduct {

  private Matrix product;
  private ClusterBasisProduct children[][];

  public ClusterBasisProduct (ClusterBasis left, ClusterBasis right) {

    int left_child = left.childrenLength(), right_child = right.childrenLength();
    boolean left_is_leaf = left_child == 0, right_is_leaf = right_child == 0;

    left_child += left_is_leaf ? 1 : 0;
    right_child += right_is_leaf ? 1 : 0;

    if (left_is_leaf && right_is_leaf) 
    { children = null; product = left.toMatrix().transpose().times(right.toMatrix()); }
    else {
      children = new ClusterBasisProduct[left_child][right_child];
      product = new Matrix(left.getRank(), right.getRank());

      for (int i = 0; i < left_child; i++) {
        Matrix Et_i = left.getTrans(i).transpose();
        ClusterBasis left_basis = left_is_leaf ? left : left.getChildren()[i];
        for (int j = 0; j < right_child; j++) {
          Matrix E_j = right.getTrans(j);
          ClusterBasis right_basis = right_is_leaf ? right : right.getChildren()[j];
          children[i][j] = new ClusterBasisProduct(left_basis, right_basis);
          product.plusEquals(Et_i.times(children[i][j].product).times(E_j));
        }
      }

    }

  }

  public Matrix getProduct () {
    return product;
  }

  public ClusterBasisProduct getChildren (int i, int j) {
    return children[i][j];
  }


}