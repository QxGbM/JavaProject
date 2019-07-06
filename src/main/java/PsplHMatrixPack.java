
import java.io.IOException;

public class PsplHMatrixPack {

  static final double epi = 1.e-10;

  @FunctionalInterface
  public interface dataFunction
  { public double body (int i, int j, int y_start, int x_start); }

  static final dataFunction testFunc = (int i, int j, int y_start, int x_start) -> 
  { return 1. / (1. + Math.abs((y_start + i) - (x_start + j))); };
  
  public static void main (String args[]) {

    int level = 6, nblocks = 2, dim = 16384, admis = 1;

    Dense d = Dense.generateDense(dim, dim, 0, 0, testFunc);

    Hierarchical h = Hierarchical.buildHMatrix(level, nblocks, dim, admis, 0, 0, testFunc);
    double compress = h.getCompressionRatio();
    System.out.println(compress);
    
    try {
      d.writeToFile("ref");
      h.writeToFile("test");
    } catch (IOException e) {
      e.printStackTrace();
    }

  }


}
