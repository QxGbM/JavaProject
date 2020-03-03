
import Jama.Matrix;

public class ClusterBasisV {

  private Matrix basis;
  private ClusterBasisV children[];

  public ClusterBasisV () {
    basis = null;
    children = null;
  }

  public ClusterBasisV (Matrix m) {
    basis = m.copy();
    children = null;
  }

  public ClusterBasisV (Dense d, int sample_rank) {
    Matrix[] rsv = d.rsvd(sample_rank);
    basis = rsv[0];
    children = null;
  }

}