package Crypto;
import java.util.ArrayList;

public class PrimitiveRoot {
	
	public static int exponent(int a, int b, int n) {
		if (b == 0) return 1;
		else return Math.floorMod((exponent(a, b-1, n) * a), n);
	}
	
	public static int discreteLog(int a, int g, int n) {
		for (int i = 0; i < n; i++) {
			if (exponent(a, i, n) == g) return i;
		}
		return -1;
	}
	
	public static ArrayList<Integer> primitiveRoots(int n) {
		int[][] list = new int[n][n];
		for (int i = 1; i <= n-1; i++) {
			for (int j = 1; j <= n-1; j++) {
				list[i][j] = exponent(i, j, n);
			}
		}
		ArrayList<Integer> roots = new ArrayList<Integer>();
		for (int i = 1; i <= n-1; i++)
			for (int j = 1; j <= n-1; j++) {
				for (int k = 1; k < j; k++) {
					if (list[i][k] == list[i][j]) {
						j = n;
						break;
					}
				}
				if (j == n-1)
					roots.add(i);
			}
		return roots;
	}
	
	public static int gcd(int a, int b) {		
		if(a == 0 || b == 0) return a + b;
		return gcd(b, Math.floorMod(a, b));
	}
	
	public static int Euler(int n) {
		int result = 0;
		for(int i = 1; i <= n; i++)
			if (gcd(n, i) == 1) result++;
		return result;
	}
	
	public static ArrayList<Integer> findN(int root) {
		ArrayList<Integer> list = new ArrayList<Integer>();
		for (int i = 1; i <= 100; i++)
			if (primitiveRoots(i).contains(root)) list.add(i);
		return list;
	}
	
	public static ArrayList<Integer> ExpField(ArrayList<Integer> list, int x, int c, int n) {
		if (n == 1) return list;
		ArrayList<Integer> list2 = new ArrayList<Integer>();
		for (int i = 0; i < list.size(); i++)
			list2.add(list.get(i) * x);
		list2.add(0);
		for (int i = 0; i < list.size(); i++)
			list.set(i, list.get(i) * c);
		list.add(0, 0);
		for (int i = 0; i < list.size(); i++)
			list.set(i, list.get(i) + list2.get(i));
		for (int i = 0; i < list.size(); i++)
			list.set(i, Math.floorMod(list.get(i), 7));
		while (list.size() >= 3) {
			int t = list.remove(0);
			t = list.get(1) - t;
			if (t < 0) t = t + 7;
			list.set(1, t);
		}
		return ExpField(list, x, c, n-1);
	}
	
	public static void main(String[] args) {
		/*System.out.println("Primitive Roots: " + primitiveRoots(229).toString());
		System.out.println("Number of Primitive Roots: " + primitiveRoots(229).size());
		System.out.println("Phi(228) = " + Euler(228));
		System.out.println("List of prime numbers having 2 as a primitive root " + findN(2));
		System.out.println("List of prime numbers having 3 as a primitive root " + findN(3));*/
		//System.out.println(discreteLog(7, 166, 433));
		ArrayList<Integer> list = new ArrayList<Integer>();
		list.add(1);list.add(1);
		System.out.println(ExpField(list, 1, 1, 1));
	}
}
