
import java.lang.System.Logger;

public class PsplHMatrixPack {

  static final double EPI = 1.e-10;
  static final int MINIMAL_SEP = 512;
  static int rank = 16;
  static int level = 3;
  static int nblocks = 2;
  static int nleaf = 128;
  static int dim = nleaf * (int) Math.pow (nblocks, level);
  static double admis = 0.5;
  static String h_name = "test";
  static String d_name = "ref";
  static boolean write_h = true;
  static boolean write_d = false;

  static final Logger logger = System.getLogger("logger");

  @FunctionalInterface
  public interface DataFunction
  { public double body (int i, int j); }

  static final DataFunction testFunc = (int i, int j) -> 
  { return 1. / (1. + Math.abs(i - j)); };

  private static void parse (String args[]) {
    StringBuilder sum = new StringBuilder();

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
      { sum.append("Ignored arg: " + args[i] + "\n"); }
    }

    sum.append("level: " + Integer.toString(level) + "\n");
    sum.append("nblocks: " + Integer.toString(nblocks) + "\n");
    sum.append("nleaf: " + Integer.toString(nleaf) + "\n");
    sum.append("dim: " + Integer.toString(dim) + "\n");
    sum.append("admis: " + Double.toString(admis) + "\n");
    sum.append("rank: " + Integer.toString(rank) + "\n");

    logger.log(System.Logger.Level.INFO, sum.toString());
  }
  
  public static void main (String args[]) {

    parse(args);

    boolean integrity = level >= 1 && nblocks >= 1 && dim >= 0 && admis >= 0;

    if (integrity) {
      Dense d = new Dense (dim, dim, 0, 0, testFunc);

      H2Matrix h2 = new H2Matrix(dim, dim, nleaf, nblocks, rank, admis, 0, 0, testFunc);
      System.out.println("compress: " + h2.toDense().minus(d).normF() / dim / dim);

      d = h2.toDense();
      long startTime = System.nanoTime();
      h2.LU();
      long endTime = System.nanoTime();
      System.out.println("H2-LU time: " +  (endTime - startTime) / 1000000);

      d.LU();
      h2.compareDense(d, "");
      System.out.println("LU: " + h2.toDense().minus(d).normF() / dim / dim);


      if (write_h) {
        Hierarchical h = new Hierarchical(dim, dim, nleaf, nblocks, admis, 0, 0, testFunc);
        double compress = h.getCompressionRatio();
        System.out.println("h Storage Compression Ratio: " + Double.toString(compress));

        System.out.print("Writing H... ");
        h.writeToFile(h_name);
        System.out.println("Done.");
      }

      if (write_d) {
        System.out.print("Writing D... ");
        d.writeToFile(d_name); 
        System.out.println("Done.");
      }
    }

  }


}
