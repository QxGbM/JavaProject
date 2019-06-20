
import java.io.IOException;

public class PsplHMatrixPack {

  @FunctionalInterface
  public interface dataFunction
  { public double body (int i, int j, int y_start, int x_start); }

  static final dataFunction testFunc = (int i, int j, int y_start, int x_start) -> 
  { return 1. / (1. + Math.abs((y_start + i) - (x_start + j))); };
  
  public static void main (String args[]) {

    int level = 1, nblocks = 2, dim = 128, admis = 1;

    Dense d = Dense.generateDense(dim, dim, 0, 0, testFunc);

    Hierarchical h = Hierarchical.buildHMatrix(level, nblocks, dim, admis, 0, 0, testFunc);
    boolean b = d.equals(h);
    System.out.println(b);
    
    try {
      h.writeToFile("test");
    } catch (IOException e) {
      e.printStackTrace();
    }

    try {
      h = Hierarchical.readFromFile("test");
      b = h.equals(d);
      System.out.println(b);

    } catch (IOException e) {
      e.printStackTrace();
    }

  }
}
