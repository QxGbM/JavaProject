package Matrix;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.LinkedList;

public interface Block {
	
	abstract public int[] getDimension();
	
	abstract public boolean isLowRank();
	
	abstract public boolean isHierarchical();
	
	abstract public Matrix[] usv();
	
	abstract public Matrix toMatrix();
	
	abstract public DenseBlock toDense();
	
	abstract public LowRankBlock toLowRank();
	
	abstract public Hierarchical toHierarchical();
	
	abstract public LinkedList<LinkedList<String>> blockString();
	
	abstract public Block[] LUdecompostion();
	
	default public void printConsole() {
		System.out.println("Matrix " + toMatrix().toString());
	}
	
	default public void printConsole(String mName) {
		System.out.println(mName + " " + toMatrix().toString());
	}
	
	default public Matrix[] rsvd(int k) {
		return SVD.rsvd(toMatrix(), k);
	}
	
	default public Block inverse() {
		if (isLowRank()) throw new ArithmeticException("No inverse for low rank blocks");
		else return new DenseBlock(toMatrix().inverse().getArrayCopy());
	}
	
	public static Block add(Block b1, Block b2) {
		if (b1.isHierarchical() || b2.isHierarchical()) {
			Hierarchical a1 = b1.toHierarchical(), a2 = b2.toHierarchical();
			Block[][] e1 = a1.getElements(), e2 = a2.getElements();
			Block e00 = add(e1[0][0], e2[0][0]);
			Block e01 = add(e1[0][1], e2[0][1]);
			Block e10 = add(e1[1][0], e2[1][0]);
			Block e11 = add(e1[1][1], e2[1][1]);
			return new Hierarchical(e00, e01, e10, e11);
		}
		else {
			return new DenseBlock(Matrix.add(b1.toMatrix(), b2.toMatrix()).getArray());
		}
	}
	
	public static Block subtract(Block b1, Block b2) {
		if (b1.isHierarchical() || b2.isHierarchical()) {
			Hierarchical a1 = b1.toHierarchical(), a2 = b2.toHierarchical();
			Block[][] e1 = a1.getElements(), e2 = a2.getElements();
			Block e00 = subtract(e1[0][0], e2[0][0]);
			Block e01 = subtract(e1[0][1], e2[0][1]);
			Block e10 = subtract(e1[1][0], e2[1][0]);
			Block e11 = subtract(e1[1][1], e2[1][1]);
			return new Hierarchical(e00, e01, e10, e11);
		}
		else {
			return new DenseBlock(Matrix.subtract(b1.toMatrix(), b2.toMatrix()).getArray());
		}
	}
	
	public static Block multiply(Block b1, Block b2) {
		if (b1.isHierarchical() || b2.isHierarchical()) {
			Hierarchical a1 = b1.toHierarchical(), a2 = b2.toHierarchical();
			Block[][] e1 = a1.getElements(), e2 = a2.getElements();
			Block e00 = add(multiply(e1[0][0], e2[0][0]), multiply(e1[0][1], e2[1][0]));
			Block e01 = add(multiply(e1[0][0], e2[0][1]), multiply(e1[0][1], e2[1][1]));
			Block e10 = add(multiply(e1[1][0], e2[0][0]), multiply(e1[1][1], e2[1][0]));
			Block e11 = add(multiply(e1[1][0], e2[0][1]), multiply(e1[1][1], e2[1][1]));
			return new Hierarchical(e00, e01, e10, e11);
		}
		else {
			if (b1.isLowRank()) {
				if (b2.isLowRank()) {
					Matrix[] usv1 = b1.usv();
					Matrix[] usv2 = b2.usv();
					return new LowRankBlock(usv1[0], 
							Matrix.multiply(Matrix.multiply(usv1[1], Matrix.multiply(usv1[2].transpose(), usv2[0])), usv2[1]),
							usv2[2]);
				}
				else {
					Matrix[] usv1 = b1.usv();
					return new LowRankBlock(usv1[0], usv1[1], Matrix.multiply(b2.toMatrix().transpose(), usv1[2]));
				}
			}
			else {
				if (b2.isLowRank()) {
					Matrix[] usv2 = b2.usv();
					return new LowRankBlock(Matrix.multiply(b1.toMatrix(), usv2[0]), usv2[1], usv2[2]);
				}
				else {
					return new DenseBlock(Matrix.multiply(b1.toMatrix(), b2.toMatrix()).getArray());
				}
			}
		}
	}
	
	default public void writeToFile (String fileName) throws IOException {
		File file = new File(fileName + ".csv");
		BufferedWriter writer = new BufferedWriter(new FileWriter(file));
		
		LinkedList<LinkedList<String>> list = blockString();
		
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
}


