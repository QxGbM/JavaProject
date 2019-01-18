package Matrix;

public class LUtest {
	
	public static int d = 5;
	public static double[][] data = {{1,1,2,3},{4,5,6,7},{8,9,10,11},{12,13,14,14}};
	
	public static Matrix a = Matrix.randomMatrix(d, d);
	//public static Matrix a = new Matrix(data);
	
	public static void testLU() {
		
		Vector b = Vector.randomVector(d, 0, 1);
		System.out.println("B:\n" + b.toString());
		
		
		Matrix[] lu = a.LUdecomposition();
		System.out.println("L:\n" + lu[0].toString());
		System.out.println("U:\n" + lu[1].toString());
		
		Matrix c = Matrix.multiply(lu[0], lu[1]);
		System.out.println("C:\n" + c.toString() + "\nC = A?: " + c.equals(a));
		
		
		Vector x = Matrix.solveLinear(a, b);
		System.out.println("X:\n" + x.toString());
		
	}
	
	public static void testInverse() {
		Matrix r = a.inverse();
		System.out.println("A-1:\n" + r.toString());
		
		Matrix ar = Matrix.multiply(a, r);
		System.out.println("A*A-1:\n" + ar.toString() + "\n" + ar.equals(Matrix.identityMatrix(d)));
	}
	
	public static void testBlockLU() {
		int[] sizes = {2,2,2,2};
		Matrix[] lu = a.blockLUdecomposition(sizes);
		System.out.println("L:\n" + lu[0].toString());
		System.out.println("U:\n" + lu[1].toString());
		
		Matrix c = Matrix.multiply(lu[0], lu[1]);
		System.out.println("C:\n" + c.toString() + "\nC = A?: " + c.equals(a));
	}
	
	public static void testSVD() {
		
		int d1 = 5, d2 = 4, rank = 5;
		
		Matrix a = Matrix.randomMatrix(d1, d2);
		Matrix b = Matrix.randomMatrix(d2, rank);
		
		a = Matrix.multiply(a, Matrix.multiply(b, b.transpose()));
		
		System.out.println("A:\n" + a.toString());
		
		Matrix[] usv = SVD.rsvd(a, 5);
		Matrix u = usv[0], s = usv[1], v = usv[2];
		
		System.out.println("U:\n" + u.toString());
		System.out.println("S:\n" + s.toString());
		System.out.println("V:\n" + v.toString());
		
		Matrix c = Matrix.multiply(Matrix.multiply(u, s), v.transpose());
		
		System.out.println("C:\n" + c.toString() + "\nC = A?: " + c.equals(a));
	}
	
	public static void testOrth() {
		System.out.println("o:\n" + SVD.orth(a).toString());
	}
	
	public static void main(String args[]) {
		/* A is a random matrix of dimension dxd
		 * L and U are the decomposed matrices of matrix A
		 * C is the product of L and U (is identical to A)
		 * B is a random vector of dimension d
		 * x is the solution of the linear system Ax = B
		 * Correctness verified
		 * */
		
		//System.out.println("A:\n" + a.toString());
        
		
		testSVD();
		

	}
}
