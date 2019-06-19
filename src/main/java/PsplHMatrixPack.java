
import java.io.IOException;

public class PsplHMatrixPack {
  
  public static void main (String args[]) {

    /*Dense d = new Dense(8, 8);
    d.plusEquals(Dense.random(8, 3).times(Dense.random(3, 8)));
    d.print(4,4);*/

    /*LowRank lr = d.toLowRank();
    boolean b = lr.equals(d);
    int r = lr.getRank();
    System.out.println(b);
    System.out.println(r);

    Hierarchical h = lr.toHierarchical(2, 2);
    b = d.equals(h);
    System.out.println(b);*/
    
    /*try {
      d.writeToFile("test");
    } catch (IOException e) {
      e.printStackTrace();
    }*/

    try {
      Dense d = Dense.readFromFile("test");
      d.print(4, 4);
    } catch (IOException e) {
      e.printStackTrace();
    }

  }
}
