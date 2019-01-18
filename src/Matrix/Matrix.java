package Matrix;

import java.util.Random;

public class Matrix {
	
	private int d0;
	private int d1;
	
	private double[][] e;
	
	public Matrix (int r, int c) {
		d0 = r; d1 = c;
		e = new double[d0][d1];
	}
	
	public Matrix (double[][] a) {
		d0 = a.length; d1 = 0;
		for (int i = 0; i < d0; i++) {
			if (a[i].length > d1) d1 = a[i].length;
		}
		e = new double[d0][d1];
		for (int i = 0; i < d0; i++) {
			for (int j = 0; j < d1 && j < a[i].length; j++) {
				e[i][j] = a[i][j];
			}
		}
	}
	
	public int[] getSize() {
		return new int[] {d0, d1};
	}
	
	public double[][] getArray() {
		return e;
	}
	
	public double[][] getArrayCopy() {
		return e.clone();
	}
	
	public String toString() {
		String s = Integer.toString(d0) + "x" + Integer.toString(d1) + "\n";
		for (int i = 0; i < d0; i++) {
			for (int j = 0; j < d1; j++) {
				s += Double.toString(e[i][j]);
				if (j != d1 - 1) {
					s += ",";
				}
			}
			if (i != d0 - 1) {
				s += "\n";
			}
		}
		return s;
	}
	
	public boolean equals(Matrix m) {
		if (this.d0 != m.d0 || this.d1 != m.d1) return false;
		for (int i = 0; i < d0; i++) {
			for (int j = 0; j < d1; j++) {
				if (Math.abs(this.e[i][j] - m.e[i][j]) > 1e-5) {
					System.out.println("c: " + i + ", " + j);
					return false;
				}
			}
		}
		return true;
	}
	
	public Matrix block(int top, int bottom, int left, int right) throws ArithmeticException {
		if (left < 0 || top < 0 || right >= d1 || bottom >= d0 || left > right || top > bottom) 
			throw new ArithmeticException("Dimension exceeds limit");
		Matrix result = new Matrix(bottom - top + 1, right - left + 1);
		for (int i = 0; i < result.d0; i++) {
			for (int j = 0; j < result.d1; j++) {
				result.e[i][j] = e[i + top][j + left];
			}
		}
		return result;
		
	}
	
	public Matrix transpose() {
		Matrix a = new Matrix(d1, d0);
		for (int i = 0; i < d0; i++) {
			for (int j = 0; j < d1; j++) {
				a.e[j][i] = e[i][j];
			}
		}
		return a;
	}
	
	public Matrix inverse() throws ArithmeticException {
		if (d0 != d1 || d0 == 0 || d1 == 0) 
			throw new ArithmeticException("Inverse does not exist.");
		Matrix a = new Matrix(e), result = identityMatrix(d0);
		for(int i = 0; i < d0; i++) {
			
			for (int j = i + 1; j < d0 && Math.abs(a.e[i][i]) < 1e-7; j++) {
				for (int k = i; k < d1 && Math.abs(a.e[j][i]) > 1e-7; k++) {
					double x = a.e[i][k]; a.e[i][k] = a.e[j][k]; a.e[j][k] = x;
					x = result.e[i][k]; result.e[i][k] = result.e[j][k]; result.e[j][k] = x;
				}
			}
			if (Math.abs(a.e[i][i]) < 1e-7) {
				throw new ArithmeticException("Determinant is 0.");
			}
			else {
				double x = a.e[i][i];
				for(int j = 0; j < d1; j++) {
					a.e[i][j] /= x;
					result.e[i][j] /= x;
				}
				for(int j = 0; j < d0; j++) {
					x = a.e[j][i];
					for(int k = 0; k < d1 && Math.abs(x) > 1e-7 && j != i; k++) {
						a.e[j][k] -= x * a.e[i][k];
						result.e[j][k] -= x * result.e[i][k];
					}
				}
			}
		}
		
		return result;
	}
	
	public Matrix[] LUdecomposition() throws ArithmeticException {
		if (d0 == 0 || d1 == 0) throw new ArithmeticException("LU decomposition does not exist.");
		Matrix[] lu = new Matrix[2];
		
		Matrix l = identityMatrix(d0), u = new Matrix(d0, d1);
		
		u.e[0][0] = e[0][0];
		
		for (int i = 0; i < d0-1 && i < d1-1; i++) {
			for (int j = i; j < d0; j++) {
				for (int k = i; k < d1; k++) {
					if (j == i) {
						if (k == i) {
							l.e[i+1][i] += e[i+1][i]; 
							l.e[i+1][i] /= u.e[i][i];
						}
						else {
							u.e[i][k] += e[i][k];
						}
					}
					else {
						if (k == i) {
							if (j == d0 - 1) {u.e[i+1][i+1] += e[i+1][i+1];}
							else {l.e[j+1][i] += e[j+1][i]; l.e[j+1][i] /= u.e[i][i];}
						}
						else {
							if (j <= k) {u.e[j][k] -= l.e[j][i] * u.e[i][k];}
							else {l.e[j][k] -= l.e[j][i] * u.e[i][k];}
						}
					}
				}
			}
		}
		
		for (int i = d1; i < d0; i++) {l.e[i][d1-1] += e[i][d1-1]; l.e[i][d1-1] /= u.e[d1-1][d1-1];}
		for (int i = d0; i < d1; i++) {u.e[d0-1][i] += e[d0-1][i];}
		
		lu[0] = l; lu[1] = u;
		return lu;
	}
	
	public Matrix[] blockLUdecomposition (int size) throws ArithmeticException {
		return blockLUdecomposition(new int[] {size});
	}
	
	public Matrix[] blockLUdecomposition (int[] sizes) throws ArithmeticException {
		if (d0 != d1 || d0 == 0 || d1 == 0)
			throw new ArithmeticException("LU decomposition does not exist.");
		else if(sizes.length == 0 || sizes[0] >= d0) {
			Matrix[] lu = new Matrix[2];
			Matrix l = identityMatrix(d0), u = new Matrix(e);
			lu[0] = l; lu[1] = u;
			return lu;
		}
		else {
			int size = sizes[0];
			Matrix[] lu = new Matrix[2];
			Matrix l = new Matrix(d0, d0), u = new Matrix(d0, d0);
			
			for (int i = 0; i < size; i++) {
				l.e[i][i] = 1.0;
				for (int j = 0; j < d0; j++) {
					u.e[i][j] = e[i][j];
				}
			}
			
			Matrix ll = multiply(this.block(size, d0 - 1, 0, size - 1), u.block(0, size - 1, 0, size - 1).inverse());
			
			for (int i = size; i < d0; i++) {
				for (int j = 0; j < size; j++) {
					l.e[i][j] = ll.e[i-size][j];
				}
			}
			
			int[] new_sizes = new int[sizes.length - 1];
			for (int i = 0; i < sizes.length - 1; i++) 
				new_sizes[i] = sizes[i + 1];
			
			Matrix[] new_lu = subtract(this.block(size, d0 - 1, size, d0 - 1), 
					multiply(l.block(size, d0 - 1, 0, size - 1), u.block(0, size - 1, size, d0 - 1)))
					.blockLUdecomposition(new_sizes);
			
			for (int i = size; i < d0; i++) {
				for (int j = size; j < d0; j++) {
					l.e[i][j] = new_lu[0].e[i-size][j-size];
					u.e[i][j] = new_lu[1].e[i-size][j-size];
				}
			}
			
			lu[0] = l; lu[1] = u;
			return lu;
		}
	}
	
	public static Matrix identityMatrix (int dimension) {
		Matrix I = new Matrix (dimension, dimension);
		for(int i = 0; i < dimension; i++) {
			I.e[i][i] = 1.0;
		}
		return I;
	}
	
	public static Matrix identityMatrix (int d0, int d1) {
		Matrix I = new Matrix (d0, d1);
		for(int i = 0; i < d0 && i < d1; i++) {
			I.e[i][i] = 1.0;
		}
		return I;
	}
	
	public static Matrix add(Matrix m1, Matrix m2) throws ArithmeticException {
		if (m1.d0 != m2.d0 || m1.d1 != m2.d1) 
			throw new ArithmeticException("Unmatched matrix dimensions.");
		
		Matrix result = new Matrix(m1.d0, m1.d1);
		
		for(int i = 0; i < result.d0; i++) {
			for (int j = 0; j < result.d1; j++) {
				result.e[i][j] = m1.e[i][j] + m2.e[i][j];
			}
		}
		
		return result;
	}
	
	public static Matrix subtract(Matrix m1, Matrix m2) throws ArithmeticException {
		if (m1.d0 != m2.d0 || m1.d1 != m2.d1) 
			throw new ArithmeticException("Unmatched matrix dimensions.");
		
		Matrix result = new Matrix(m1.d0, m1.d1);
		
		for(int i = 0; i < result.d0; i++) {
			for (int j = 0; j < result.d1; j++) {
				result.e[i][j] = m1.e[i][j] - m2.e[i][j];
			}
		}
		
		return result;
	}
	
	public static Matrix multiply(Matrix m1, Matrix m2) throws ArithmeticException {
		if (m1.d1 != m2.d0) 
			throw new ArithmeticException("Unmatched matrix dimensions. " + m1.d1 + " and " + m2.d0);
		
		Matrix result = new Matrix(m1.d0, m2.d1);
		
		for(int i = 0; i < result.d0; i++) {
			for (int j = 0; j < result.d1; j++) {
				for (int k = 0; k < m1.d1; k++) {
					result.e[i][j] += m1.e[i][k] * m2.e[k][j];
				}
			}
		}
		
		return result;
	}
	
	public static Matrix randomMatrix(int r, int c) {
		Matrix m = new Matrix(r, c);
		Random rand = new Random();
		for (int i = 0; i < m.d0; i++) {
			for (int j = 0; j < m.d1; j++) {
				m.e[i][j] = rand.nextGaussian();
			}
		}
		
		return m;
	}
	
	public static Vector solveLinear(Matrix m, Vector b) throws ArithmeticException {
		if (m.d0 != b.dimension()) throw new ArithmeticException("Unmatch dimension");
		
		Matrix[] lu = m.LUdecomposition();
		Matrix l = lu[0], u = lu[1];
		int d = m.d1;
		
		double[] z = new double[d];
		for (int i = 0; i < d; i++) {
			z[i] = b.elements()[i];
			for (int j = 0; j < i; j++) {
				z[i] -= l.e[i][j] * z[j];
			}
		}
		
		double[] x = new double[d];
		for (int i = d-1; i >= 0; i--) {
			x[i] = z[i];
			for (int j = d-1; j > i; j--) {
				x[i] -= u.e[i][j] * x[j];
			}
			x[i] /= u.e[i][i];
		}
		
		return new Vector(x);
	}
	
	public static Matrix solveLinearSystems_Left(Matrix a, Matrix b) {
		if (a.d0 != b.d0) throw new ArithmeticException("Unmatch dimension");
		
		Matrix[] lu = a.LUdecomposition();
		Matrix l = lu[0], u = lu[1];
		
		Matrix c = new Matrix(a.d0, b.d1);
		for (int i = 0; i < a.d0; i++) {
			for (int j = 0; j < b.d1; j++) {
				c.e[i][j] = b.e[i][j];
				for (int k = 0; k < i; k++) {
					c.e[i][j] -= l.e[i][k] * c.e[k][j];
				}
			}
		}
		
		Matrix d = new Matrix(a.d1, b.d1);
		for (int i = a.d1 - 1; i >= 0; i--) {
			for (int j = 0; j < b.d1; j++) {
				d.e[i][j] = c.e[i][j];
				for (int k = i + 1; k < a.d1; k++) {
					d.e[i][j] -= d.e[k][j] * u.e[i][k];
				}
				d.e[i][j] /= u.e[i][i];
			}
		}
		
		return d;
	}
	
	public static Matrix solveLinearSystems_Right(Matrix a, Matrix b) {
		if (a.d1 != b.d1) throw new ArithmeticException("Unmatch dimension");
		
		Matrix[] lu = a.LUdecomposition();
		Matrix l = lu[0], u = lu[1];
		
		Matrix c = new Matrix(b.d0, a.d0);
		for (int i = 0; i < b.d0; i++) {
			for (int j = 0; j < a.d0; j++) {
				c.e[i][j] = b.e[i][j];
				for (int k = 0; k < j; k++) {
					c.e[i][j] -= c.e[i][k] * u.e[k][j];
				}
				c.e[i][j] /= u.e[j][j];
			}
		}
		
		for (int i = 0; i < b.d0; i++) {
			for (int j = a.d0 - 1; j >= 0; j--) {
				for (int k = a.d0 - 1; k > j; k--) {
					c.e[i][j] -= c.e[i][k] * l.e[k][j]; 
				}
			}
		}
		
		
		return c;
	}
}
