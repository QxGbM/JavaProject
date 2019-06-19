
import java.io.IOException;

public class PsplHMatrixPack {
  
  public static void main (String args[]) {

    Dense d = new Dense(8, 8);
    d.plusEquals(Dense.random(8, 3).times(Dense.random(3, 8)));

    LowRank lr = d.toLowRank();
    boolean b = lr.equals(d);
    int r = lr.getRank();
    System.out.println(b);
    System.out.println(r);

    Hierarchical h = lr.toHierarchical(2, 2);
    b = d.equals(h);
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
