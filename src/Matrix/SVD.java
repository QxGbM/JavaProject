package Matrix;

public class SVD {
	
	public static Matrix[] rsvd (Matrix A, int k) {
		int m = A.getSize()[0], n = A.getSize()[1], p = Math.min(2 * k, n);
		if (A.equals(new Matrix(m, n))) {
			return new Matrix[] {new Matrix(m, 0), new Matrix(0, 0), new Matrix(n, 0)};
		}
		else {
			Matrix X = Matrix.randomMatrix(n, p);
			Matrix Y = Matrix.multiply(A, X);
			Matrix w1 = orth(Y);
			Matrix B = Matrix.multiply(w1.transpose(), A).transpose();
			
			Jama.Matrix b = new Jama.Matrix(B.getArrayCopy());
			Jama.SingularValueDecomposition s = b.svd();
			Matrix w2 = new Matrix(s.getV().getArrayCopy());
			Matrix S = new Matrix(s.getS().getArrayCopy());
			Matrix V = new Matrix(s.getU().getArrayCopy());
			
			Matrix U = Matrix.multiply(w1, w2);
			k = Math.min(k, U.getSize()[1]);
			
			Matrix[] usv = new Matrix[3];
			usv[0] = U.block(0, U.getSize()[0] - 1, 0, k - 1);
			usv[1] = S.block(0, k - 1, 0, k - 1);
			usv[2] = V.block(0, V.getSize()[0] - 1, 0, k - 1);
			
			return usv;
		}
	}
	
	public static Matrix orth (Matrix A) {
		Jama.Matrix B = new Jama.Matrix(A.getArrayCopy());
		Jama.SingularValueDecomposition s = B.svd();
		double[][] data = s.getU().getMatrix(0, A.getSize()[0] - 1, 0, s.rank() - 1).getArrayCopy();
		return new Matrix(data);
	}

}