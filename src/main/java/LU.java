
public class LU {
  
  public static void main (String args[]) {

    Dense d = new Dense(16, 16);
    d.plusEquals(Dense.random(16, 16));

    LowRank lr = d.toLowRank();
    boolean b = lr.equals(d);
    System.out.println(b);

  }
}
