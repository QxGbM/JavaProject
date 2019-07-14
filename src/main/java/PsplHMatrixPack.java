
import java.io.IOException;

public class PsplHMatrixPack {

  static final double epi = 1.e-10;

  @FunctionalInterface
  public interface dataFunction
  { public double body (int i, int j, int y_start, int x_start); }

  static final dataFunction testFunc = (int i, int j, int y_start, int x_start) -> 
  { return 1. / (1. + Math.abs((y_start + i) - (x_start + j))); };
  
  public static void main (String args[]) {

    int level = 3, nblocks = 2, dim = 2048, admis = 1;
    
    String h_name = "test", d_name = "ref";
    boolean write_h = true, write_d = true;

    for (int i = 0; i < args.length; i++)
    {
      if (args[i].startsWith("-level="))
      { level = Integer.parseInt(args[i].substring(7)); }
      else if (args[i].startsWith("-nblocks="))
      { nblocks = Integer.parseInt(args[i].substring(9)); }
      else if (args[i].startsWith("-dim="))
      { dim = Integer.parseInt(args[i].substring(5)); }
      else if (args[i].startsWith("-admis="))
      { admis = Integer.parseInt(args[i].substring(7)); }
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
    System.out.println("Level: " + Integer.toString(level));
    System.out.println("nblocks: " + Integer.toString(nblocks));
    System.out.println("dim: " + Integer.toString(dim));
    System.out.println("admis: " + Integer.toString(admis));

    boolean integrity = level >= 0 && nblocks >= 1 && dim >= 0 && admis >= 0;

    if (integrity && (write_d || write_h))
    try {
      Dense d = Dense.generateDense(dim, dim, 0, 0, testFunc);

      if (write_h)
      {
        Hierarchical h = Hierarchical.buildHMatrix(level, nblocks, dim, admis, 0, 0, testFunc);
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
