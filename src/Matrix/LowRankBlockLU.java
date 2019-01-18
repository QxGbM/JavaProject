package Matrix;

import java.io.IOException;

public class LowRankBlockLU {
	
	public static Matrix testMatrix() {
		Matrix a = new Matrix(200, 200);
		int[] size = a.getSize();
		double[][] data = a.getArray();
		for(int i = 0; i < size[0]; i++) {
			for(int j = 0; j < size[1]; j++) {
				data[i][j] = 1 / (double)(Math.abs(i-j) + 1);
			}
		}
		return a;
	}
	
	public static void main(String[] args) throws IOException {

		Matrix a = testMatrix();

		Block h = new Hierarchical(a, 15);
		h.writeToFile("b");
		
		Block[] lu = h.LUdecompostion();
		lu[0].writeToFile("l");
		lu[1].writeToFile("u");
		
		System.out.println(a.equals(Matrix.multiply(lu[0].toMatrix(), lu[1].toMatrix())));
		
	}
	
}
