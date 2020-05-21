
import java.io.IOException;

public class PsplHMatrixPack {

  static final double epi = 1.e-10;
  static final int minimal_sep = 512;
  static int rank = 16;

  @FunctionalInterface
  public interface dataFunction
  { public double body (int i, int j); }

  static final dataFunction testFunc = (int i, int j) -> 
  { return 1. / (1. + Math.abs(i - j)); };
  
  public static void main (String args[]) {

    int level = 1, nblocks = 4, nleaf = 256, dim = nleaf * (int) Math.pow (nblocks, level);
    double admis = 0.5;
    
    String h_name = "test", d_name = "ref";
    boolean write_h = true, write_d = false;

    for (int i = 0; i < args.length; i++)
    {
      if (args[i].startsWith("-level="))
      { level = Integer.parseInt(args[i].substring(7)); }
      else if (args[i].startsWith("-nblocks="))
      { nblocks = Integer.parseInt(args[i].substring(9)); }
      else if (args[i].startsWith("-nleaf="))
      { nleaf = Integer.parseInt(args[i].substring(7)); dim = nleaf * (int) Math.pow (nblocks, level); }
      else if (args[i].startsWith("-dim="))
      { dim = Integer.parseInt(args[i].substring(5)); nleaf = dim / (int) Math.pow (nblocks, level); }
      else if (args[i].startsWith("-admis="))
      { admis = Integer.parseInt(args[i].substring(7)); }
      else if (args[i].startsWith("-rank="))
      { rank = Integer.parseInt(args[i].substring(6)); }
      else if (args[i].startsWith("-h="))
      { write_h = true; h_name = args[i].substring(3); }
      else if (args[i].startsWith("-d="))
      { write_d = true; d_name = args[i].substring(3); }
      else if (args[i].startsWith("-skiph"))
      { write_h = false; }
      else if (args[i].startsWith("-skipd"))
      { write_d = false; }
      else 
      { System.out.println("Ignored arg: " + args[i]); }
    }

    System.out.println("Running Summary: ");
    System.out.println("level: " + Integer.toString(level));
    System.out.println("nblocks: " + Integer.toString(nblocks));
    System.out.println("nleaf: " + Integer.toString(nleaf));
    System.out.println("dim: " + Integer.toString(dim));
    System.out.println("admis: " + Double.toString(admis));
    System.out.println("rank: " + Integer.toString(rank));

    boolean integrity = level >= 1 && nblocks >= 1 && dim >= 0 && admis >= 0;

    if (integrity)
    try {
      Dense d = new Dense (dim, dim, 0, 0, testFunc);

      /*H2Matrix h2 = new H2Matrix(dim, dim, nleaf, nblocks, rank, admis, 0, 0, testFunc);
      System.out.println("compress: " + h2.toDense().minus(d).normF() / dim / dim);
      System.out.println(h2.structure());

      long startTime = System.nanoTime();
      h2.LU();
      long endTime = System.nanoTime();
      System.out.println("BLR-LU time: " +  (endTime - startTime) / 1000000);

      d.LU();
      h2.compareDense(d);
      System.out.println("LU: " + h2.toDense().minus(d).normF() / dim / dim);*/

      H2Matrix h2_test = new H2Matrix(1024, 1024, 128, 2, 16, 0.3, 0, 0, testFunc);
      H2Matrix h2_01 = h2_test.getElement(0, 1).castH2Matrix();//.getElement(0, 1).castH2Matrix();
      H2Matrix h2_10 = h2_test.getElement(1, 0).castH2Matrix();//.getElement(1, 0).castH2Matrix();

      ClusterBasis col = h2_10.getColBasis();
      Jama.Matrix test = col.h2matrixTimes(h2_01);

      Jama.Matrix ref = h2_01.toDense().times(col.toMatrix());
      System.out.println(h2_01.structure());

      System.out.println("test: " + ref.minus(test).normF() / ref.getRowDimension() / ref.getColumnDimension());


      if (write_h)
      {
        Hierarchical h = new Hierarchical(dim, dim, nleaf, nblocks, admis, 0, 0, testFunc);
        double compress = h.getCompressionRatio();
        System.out.println("h Storage Compression Ratio: " + Double.toString(compress));

        System.out.print("Writing H... ");
        h.writeToFile(h_name);
        System.out.println("Done.");
      }

      if (write_d)
      {
        System.out.print("Writing D... ");
        d.writeToFile(d_name); 
        System.out.println("Done.");
      }

    } catch (IOException e) {
      e.printStackTrace();
    }

  }


}
