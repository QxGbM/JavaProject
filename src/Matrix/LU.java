package Matrix;

public class LU {
	
	public static void main(String args[]) {
		/* A is a random matrix of dimension dxd
		 * L and U are the decomposed matrices of matrix A
		 * C is the product of L and U (is identical to A)
		 * B is a random vector of dimension d
		 * x is the solution of the linear system Ax = B
		 * Correctness verified
		 * */
		double[][] data1 = {{1,1,2,3},{4,5,6,7},{8,9,10,11},{12,13,14,14}};
		double[][] data2 = //{{1},{2},{3},{4}};
			                //{{1, 2, 3, 4}};
			{{1,1,2,3},{4,5,6,7},{8,9,10,11},{12,13,14,14}};
		Matrix a = new Matrix(data1);
		Matrix b = new Matrix(data2);
		System.out.println("A:\n" + a.toString() + "\nB:\n" + b.toString());
		
		Matrix[] lu = a.LUdecomposition();
		System.out.println("L:\n" + lu[0].toString());
		System.out.println("U:\n" + lu[1].toString());
		
		Matrix d = Matrix.solveLinearSystems_Right(a, b);
		System.out.println("D:\n" + d.toString() + "\n" + Matrix.multiply(d, a).toString());
		
	}
}
