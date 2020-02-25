
import java.io.IOException;

public class PsplHMatrixPack {

  static final double epi = 1.e-10;

  static int rank = 16;

  @FunctionalInterface
  public interface dataFunction
  { public double body (int i, int j, int y_start, int x_start); }

  static final dataFunction testFunc = (int i, int j, int y_start, int x_start) -> 
  { return 1. / (1. + Math.abs((y_start + i) - (x_start + j))); };
  
  public static void main (String args[]) {

    int level = 2, nblocks = 2, nleaf = 256, nleaf_max = 0, dim = nleaf * (int) Math.pow (nblocks, level), admis = 1;
    
    String h_name = "test", d_name = "ref";
    boolean write_h = true, write_d = true;

    for (int i = 0; i < args.length; i++)
    {
      if (args[i].startsWith("-level="))
      { level = Integer.parseInt(args[i].substring(7)); }
      else if (args[i].startsWith("-nblocks="))
      { nblocks = Integer.parseInt(args[i].substring(9)); }
      else if (args[i].startsWith("-nleaf="))
      { nleaf = Integer.parseInt(args[i].substring(7)); dim = nleaf * (int) Math.pow (nblocks, level); }
      else if (args[i].startsWith("-nleaf_max="))
      { nleaf_max = Integer.parseInt(args[i].substring(11)); }
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
    System.out.println("nleaf_max: " + Integer.toString(nleaf_max));
    System.out.println("dim: " + Integer.toString(dim));
    System.out.println("admis: " + Integer.toString(admis));
    System.out.println("rank: " + Integer.toString(rank));

    boolean integrity = level >= 1 && nblocks >= 1 && dim >= 0 && admis >= 0;
    if (nleaf_max < nleaf)
    { nleaf_max = dim; }

    if (integrity && (write_d || write_h))
    try {
      Dense d = new Dense (dim, dim, 0, 0, testFunc);

      /*UniformHierarchical uh = new UniformHierarchical(d, 2, 2, 16, 64);
      uh.print(0,3);

      System.out.println(uh.toDense().minus(d).normF() / dim / dim);*/

      if (write_h)
      {
        Hierarchical h = Hierarchical.buildHMatrix(level - 1, nblocks, nleaf, nleaf_max, admis, 0, 0, testFunc);
        double compress = h.getCompressionRatio();
        System.out.println("Storage Compression Ratio: " + Double.toString(compress));

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
