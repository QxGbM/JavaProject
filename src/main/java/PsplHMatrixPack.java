
import java.io.IOException;

public class PsplHMatrixPack {

  static final double epi = 1.e-10;

  static int rank = 16;

  @FunctionalInterface
  public interface dataFunction
  { public double body (int i, int j); }

  static final dataFunction testFunc = (int i, int j) -> 
  { return 1. / (1. + Math.abs(i - j)); };
  
  public static void main (String args[]) {

    int level = 4, nblocks = 2, nleaf = 128, nleaf_max = 0, dim = nleaf * (int) Math.pow (nblocks, level);
    double admis = 0.5;
    
    String h_name = "test", d_name = "ref";
    boolean write_h = false, write_d = false;

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
    System.out.println("admis: " + Double.toString(admis));
    System.out.println("rank: " + Integer.toString(rank));

    boolean integrity = level >= 1 && nblocks >= 1 && dim >= 0 && admis >= 0;
    if (nleaf_max < nleaf)
    { nleaf_max = dim; }

    if (integrity)
    try {
      Dense d = new Dense (dim, dim, 0, 0, testFunc);

      /*Dense d2 = new Dense (64, 64, 0, 2048, testFunc);
      LowRank lr1 = d2.toLowRank();
      Jama.Matrix u = Dense.getBasisU(0, 64, 16, 0.9, testFunc);
      Jama.Matrix vt = Dense.getBasisVT(2048, 64, 16, 0.9, testFunc);
      lr1.useBasis(u, vt);
      Dense d3 = lr1.toDense();
      System.out.println("error:" + d3.minusEquals(d2).normF() / 64 / 64);*/

      UniformBLR blr = new UniformBLR(dim, dim, nleaf, 0, 0, rank, admis, testFunc);
      ClusterBasis cbu = new ClusterBasis(0, dim, true, nleaf, nblocks, rank, admis, testFunc);
      cbu.convertReducedStorageForm();

      System.out.println(blr.toDense().minus(d).normF() / dim / dim);

      H2Matrix h2 = new H2Matrix(dim, dim, nleaf, nblocks, rank, admis, 0, 0, testFunc);

      System.out.println(h2.toDense().minus(d).normF() / dim / dim);


      if (write_h)
      {
        Hierarchical h = new Hierarchical(dim, dim, nleaf, nblocks, admis, 0, 0, testFunc);
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
