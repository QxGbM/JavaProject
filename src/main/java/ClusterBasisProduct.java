
import Jama.Matrix;

public class ClusterBasisProduct {

  private Matrix product;
  private ClusterBasisProduct children[][];

  public ClusterBasisProduct (ClusterBasis left, ClusterBasis right) {

    int left_child = left.childrenLength(), right_child = right.childrenLength();
    boolean left_is_leaf = left_child == 0, right_is_leaf = right_child == 0;

    left_child += left_is_leaf ? 1 : 0;
    right_child += right_is_leaf ? 1 : 0;

    children = new ClusterBasisProduct[left_child][right_child];

    for (int i = 0; i < left_child; i++) {
      for (int j = 0; j < right_child; j++) {
        children[i][j] = new ClusterBasisProduct(left, right); // TODO
      }
    }

    if (left.getRow_Col() && right.getRow_Col()) {

    }

  }


}