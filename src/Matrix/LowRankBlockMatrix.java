package Matrix;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.LinkedList;

public class LowRankBlockMatrix {
	
	private LinkedList<Integer> rows;
	private LinkedList<Integer> columns;
	
	private int totalRows;
	private int totalColumns;
	
	private Block[][] e;
	
	private int k;
	
	public LowRankBlockMatrix (Matrix a, int[] rowSizes, int[] columnSizes, int k) {
		totalRows = a.getSize()[0];
		totalColumns = a.getSize()[1];
		rows = new LinkedList<Integer>();
		columns = new LinkedList<Integer>();
		this.k = k;
		
		int sum = 0;
		for (int i = 0; i < rowSizes.length && sum < totalRows; i++) {
			if (rowSizes[i] > 0 && sum + rowSizes[i] < totalRows) {
				rows.add(rowSizes[i]);
				sum += rowSizes[i];
			}
		}
		rows.add(totalRows - sum);
		
		sum = 0;
		for (int i = 0; i < columnSizes.length && sum < totalColumns; i++) {
			if (columnSizes[i] > 0 && sum + columnSizes[i] < totalColumns) {
				columns.add(columnSizes[i]);
				sum += columnSizes[i];
			}
		}
		columns.add(totalColumns - sum);
		e = new Block[rows.size()][columns.size()];
		
		int currentRow = 0;
		
		for (int i = 0; i < rows.size(); i++) {
			int currentCol = 0;
			for (int j = 0; j < columns.size(); j++) {
				Matrix m = a.block(currentRow, currentRow + rows.get(i) -1, currentCol, currentCol + columns.get(j) -1);
				if (i == j) e[i][j] = new DenseBlock(m.getArray());
				else e[i][j] = new LowRankBlock(m, k);
				currentCol += columns.get(j);
			}
			currentRow += rows.get(i);
		}

	}
	
	private LowRankBlockMatrix (Matrix a, LinkedList<Integer> rows, LinkedList<Integer> columns, int k) {
		totalRows = a.getSize()[0];
		totalColumns = a.getSize()[1];
		this.rows = new LinkedList<Integer>();
		this.columns = new LinkedList<Integer>();
		this.k = k;

		this.rows.addAll(rows);
		this.columns.addAll(columns);

		e = new Block[rows.size()][columns.size()];
		
		int currentRow = 0;
		
		for (int i = 0; i < rows.size(); i++) {
			int currentCol = 0;
			for (int j = 0; j < columns.size(); j++) {
				Matrix m = a.block(currentRow, currentRow + rows.get(i) -1, currentCol, currentCol + columns.get(j) -1);
				if (i == j) e[i][j] = new DenseBlock(m.getArray());
				else e[i][j] = new LowRankBlock(m, k);
				currentCol += columns.get(j);
			}
			currentRow += rows.get(i);
		}
	}
	
	private LowRankBlockMatrix (Block[][] e, LinkedList<Integer> rows, LinkedList<Integer> columns, int totalRows, int totalColumns, int k) {
		this.e = e;
		this.rows = new LinkedList<Integer>();
		this.columns = new LinkedList<Integer>();
		this.rows.addAll(rows);
		this.columns.addAll(columns);
		this.totalRows = totalRows;
		this.totalColumns = totalColumns;
		this.k = k;
	}
	
	public void writeToFile(String fileName) throws IOException {
		File file = new File(fileName + ".csv");
		BufferedWriter writer = new BufferedWriter(new FileWriter(file));
		
		LinkedList<LinkedList<String>> list = new LinkedList<LinkedList<String>>();
		int row = 0;
		
		for (int i = 0; i < rows.size(); i++) {
			
			for (int j = 0; j < columns.size(); j++) {
				
				Matrix block = e[i][j].toMatrix();
				double[][] data = block.getArrayCopy();
				int[] size = block.getSize();
				
				while (list.size() < row + size[0] + 1) list.add(new LinkedList<String>());
				
				list.get(row).add("Block[" + i + "/" + j + "]");
				list.get(row).add(size[0] + "x" + size[1]);
				
				for (int k = 0; k < size[0]; k++) {
					for (int l = 0; l < size[1]; l++) {
						String s = Double.toString(data[k][l]);
						list.get(row + k + 1).add(s);
					}
				}
				
				if (j != columns.size() - 1) {
					int n = 0;
					for (int k = 0; k <= size[0]; k++) {
						if (list.get(row + k).size() > n) n = list.get(row + k).size();
					}
					
					for (int k = 0; k <= size[0]; k++) {
						while (list.get(row + k).size() <= n) list.get(row + k).add("");
					}
				}
			}
			
			if (i != rows.size() - 1) {
				while (list.size() > row && list.get(row).size() > 0) {
					row++;
				}
				row++; list.add(new LinkedList<String>());
			}
		}
		
		for (int i = 0; i < list.size(); i++) {
			for (int j = 0; j < list.get(i).size(); j++) {
				writer.write(list.get(i).get(j));
				if (j != list.get(i).size() - 1) writer.write(",");
			}
			writer.newLine();
		}
		
		writer.flush();
		writer.close();
	}
	
	public LowRankBlockMatrix[] LUdecomposition () throws ArithmeticException {
		
		LowRankBlockMatrix l = new LowRankBlockMatrix(Matrix.identityMatrix(totalRows), rows, rows, k);
		if (rows.size() == 1 || columns.size() == 1) {
			return new LowRankBlockMatrix[] {l, this};
		}
		else {
			LowRankBlockMatrix u = new LowRankBlockMatrix(new Matrix(totalRows, totalColumns), rows, columns, k);
			
			for (int i = 0; i < columns.size(); i++) {
				u.e[0][i] = e[0][i];
			}
			
			for (int i = 1; i < rows.size(); i++) {
				l.e[i][0] = Block.multiply(e[i][0], u.e[0][0].inverse());
			}
			
			LinkedList<Integer> new_rows = new LinkedList<Integer>();
			new_rows.addAll(rows);
			int r = new_rows.removeFirst();
			
			LinkedList<Integer> new_columns = new LinkedList<Integer>();
			new_columns.addAll(columns);
			int c = new_columns.removeFirst();
			
			Block[][] new_e = new Block[rows.size()-1][columns.size()-1];
			for (int i = 0; i < rows.size()-1; i++) {
				for (int j = 0; j < columns.size()-1; j++) {
					new_e[i][j] = Block.subtract(e[i+1][j+1], Block.multiply(l.e[i+1][0], u.e[0][j+1]));
				}
			}
			
			LowRankBlockMatrix sub = new LowRankBlockMatrix(new_e, new_rows, new_columns, totalRows-r, totalColumns-c, k);
			LowRankBlockMatrix[] new_lu = sub.LUdecomposition();
			
			for (int i = 1; i < rows.size(); i++) {
				for (int j = 1; j < rows.size(); j++) {
					l.e[i][j] = new_lu[0].e[i-1][j-1];
				}
				for (int j = 1; j < columns.size(); j++) {
					u.e[i][j] = new_lu[1].e[i-1][j-1];
				}
			}
			
			LowRankBlockMatrix[] lu = new LowRankBlockMatrix[2];
			lu[0] = l; lu[1] = u;
			return lu;
		}
	}
	
	public Matrix toMatrix() {
		Matrix A = new Matrix(totalRows, totalColumns);
		double[][] data = A.getArray();
		int currentRow = 0;
		
		for(int i = 0; i < rows.size(); i++) {
			int currentColumn = 0;
			for(int j = 0; j < columns.size(); j++) {
				Matrix a = e[i][j].toMatrix();
				double[][] array = a.getArray();
				int m = a.getSize()[0], n = a.getSize()[1];
				for(int k = 0; k < m; k++) {
					for(int l = 0; l < n; l++) {
						data[currentRow+k][currentColumn+l] = array[k][l];
					}
				}
				currentColumn += columns.get(j);
			}
			currentRow += rows.get(i);
		}
		return A;
	}
}
