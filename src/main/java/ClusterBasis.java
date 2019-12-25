
import Jama.Matrix;

public class ClusterBasis {

  private Matrix elements[];
  private ClusterBasis children[];
  
  public ClusterBasis (int n, int dim) {
    elements = new Matrix[n];
    children = null;
    for (int i = 0; i < n; i++)
    { elements[i] = new Matrix(dim, dim); }
  }

  public ClusterBasis (int n, int dim, int n_child) {
    elements = new Matrix[n];
    children = new ClusterBasis[n_child];
    for (int i = 0; i < n; i++)
    { elements[i] = new Matrix(dim, dim); }
  }

  public void setElement (int index, Matrix m) {
    elements[index] = m.copy();
  }

  public void setChildren (int index, ClusterBasis b) {
    children[index] = b;
  }

  public void print (int w, int d) {
    for (int i = 0; i < elements.length; i++)
    {
      System.out.println("B"+i);
      elements[i].print(w, d);
    }
  }

}