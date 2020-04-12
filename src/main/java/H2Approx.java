
import Jama.Matrix;

public class H2Approx {
  private Matrix S;
  private H2Approx children[][];

  public H2Approx (Matrix S) {
    this.S = S;
    children = null;
  }

  public H2Approx (int m, int n) {
    S = null;
    children = new H2Approx[m][n];
  }

  public H2Approx (ClusterBasis left_prime, ClusterBasis right_prime, ClusterBasisProduct X, ClusterBasisProduct Y, H2Matrix h) {
    if (left_prime.childrenLength() == 0 || right_prime.childrenLength() == 0) {
      S = left_prime.toMatrix().transpose().times(h.toDense()).times(right_prime.toMatrix());
      children = null;
    }
    else {
      int m = h.getNRowBlocks(), n = h.getNColumnBlocks();
      children = new H2Approx[m][n];

      for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
          Block b = h.getElement(i, j);
          if (b.getType() == Block.Block_t.LOW_RANK) {
            Matrix product = X.getProduct(i).times(b.toLowRank().getS()).times(Y.getProduct(j));
            children[i][j] = new H2Approx(product);
          } 
          else if (b.getType() == Block.Block_t.DENSE) {
            Matrix product = left_prime.toMatrix(i).transpose().times(b.toDense()).times(right_prime.toMatrix(j));
            children[i][j] = new H2Approx(product);
          }
          else {
            children[i][j] = new H2Approx(left_prime.getChildren()[i], right_prime.getChildren()[j], X.getChildren(i), Y.getChildren(j), b.castH2Matrix());
          }
        }
      }
      S = collectProduct(left_prime, right_prime);
    }
  }

  public int getNRowBlocks()
  { return children.length; }

  public int getNColumnBlocks()
  { return children[0].length; }

  public Matrix getS () {
    return S;
  }

  public Matrix getProduct (int i, int j) {
    return getChildren(i, j) == null ? null : children[i][j].S;
  }

  public void accumProduct (Matrix product) {
    if (this.S == null)
    { this.S = new Matrix(product.getArray()); }
    else 
    { this.S.plusEquals(product); }
  }

  public void accumProduct (int i, int j, Matrix product) {
    if (children[i][j] != null)
    { children[i][j].accumProduct(product); }
    else
    { children[i][j] = new H2Approx(product); }
  }

  public H2Approx expandChildren (int i, int j, int m, int n) {
    if (children[i][j] == null)
    { children[i][j] = new H2Approx(m, n); }
    else
    { children[i][j].children = new H2Approx[m][n]; }
    return children[i][j];
  }

  public Matrix collectProduct (ClusterBasis left, ClusterBasis right) {
    Matrix product = new Matrix (left.getRank(), right.getRank());
    for (int i = 0; i < getNRowBlocks(); i++) {
      Matrix Et_i = left.getTrans(i).transpose();
      for (int j = 0; j < getNColumnBlocks(); j++) {
        Matrix E_j = right.getTrans(j);
        product.plusEquals(Et_i.times(children[i][j].S).times(E_j));
      }
    }
    return product;
  }

  public void splitProduct (ClusterBasis left, ClusterBasis right) {
    if (S != null && children != null)
    for (int i = 0; i < getNRowBlocks(); i++) {
      Matrix E_i = left.getTrans(i).times(S);
      for (int j = 0; j < getNColumnBlocks(); j++) {
        Matrix Et_j = right.getTrans(j).transpose();
        accumProduct(i, j, E_i.times(Et_j));
      }
    }
    S = null;
  }

  public H2Approx getChildren (int i, int j) {
    return children == null ? null : children[i][j];
  }


}