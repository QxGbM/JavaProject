package Matrix;

import java.util.LinkedList;

public class LowRankBlock implements Block{
	
	private double[][] e;
	
	private Matrix[] usv = new Matrix[3];
	
	public LowRankBlock (Matrix a, int k) {
		e = a.getArrayCopy();
		usv = rsvd(k);
	}
	
	public LowRankBlock (Matrix u, Matrix s, Matrix v) {
		usv[0] = u;
		usv[1] = s;
		usv[2] = v;
	}
	
	@Override
	public int[] getDimension() {
		return new int[] {usv[0].getSize()[0], usv[2].getSize()[0]};
	}
	
	@Override
	public boolean isLowRank() {
		return true;
	}
	
	@Override
	public boolean isHierarchical() {
		return false;
	}
	
	@Override
	public Matrix[] usv() {
		return usv;
	}
	
	@Override
	public Matrix toMatrix() {
		if (e == null) {
			Matrix m = Matrix.multiply(Matrix.multiply(usv[0], usv[1]), usv[2].transpose());
			e = m.getArrayCopy();
			return m;
		}
		else {
			return new Matrix(e);
		}
	}
	
	@Override
	public DenseBlock toDense() {
		return new DenseBlock(toMatrix().getArray());
	}

	@Override
	public LowRankBlock toLowRank() {
		return this;
	}

	@Override
	public Hierarchical toHierarchical() {
		int sizeu = usv[0].getSize()[0], sizev = usv[2].getSize()[0], rank = usv[1].getSize()[0];
		int m = sizeu / 2, n = sizev / 2;
		Block e00 = new LowRankBlock(usv[0].block(0, m - 1, 0, rank - 1), usv[1], usv[2].block(0, n - 1, 0, rank - 1));
		Block e01 = new LowRankBlock(usv[0].block(0, m - 1, 0, rank - 1), usv[1], usv[2].block(n, sizev - 1, 0, rank - 1));
		Block e10 = new LowRankBlock(usv[0].block(m, sizeu - 1, 0, rank - 1), usv[1], usv[2].block(0, n - 1, 0, rank - 1));
		Block e11 = new LowRankBlock(usv[0].block(m, sizeu - 1, 0, rank - 1), usv[1], usv[2].block(n, sizev - 1, 0, rank - 1));
		return new Hierarchical(e00, e01, e10, e11);
	}
	
	/*
	
	@Override // Outputs in dense block
	public LinkedList<LinkedList<String>> blockString() {
		return new DenseBlock(toMatrix().getArray()).blockString();
	}
	
	//*/
	
	// /*
	
	@Override // Outputs in usv
	public LinkedList<LinkedList<String>> blockString() {
		LinkedList<LinkedList<String>> list = new LinkedList<LinkedList<String>>();
		
		list.add(new LinkedList<String>());
		list.get(0).add("LowRankBlock");
		list.get(0).add("rank:" + Math.min(usv[1].getSize()[0], usv[1].getSize()[1]));
		list.get(0).add(usv[0].getSize()[0] + "x" + usv[2].getSize()[0]);
		
		int d0 = usv[0].getSize()[0], d1 = usv[0].getSize()[1];
		double[][] data = usv[0].getArray();
		list.add(new LinkedList<String>());
		list.get(1).add("u");
		list.get(1).add(d0 + "x" + d1);
		
		for(int i = 0; i < d0; i++) {
			list.add(new LinkedList<String>());
			for(int j = 0; j < d1; j++) {
				list.get(i + 2).add(Double.toString(data[i][j]));
			}
		}
		
		for(int i = 0; i < list.size(); i++) {
			while (list.get(i).size() <= d1) list.get(i).add("");
		}
		
		int m = d0+3, n = d1;
		
		d0 = usv[1].getSize()[0];
		d1 = usv[1].getSize()[1];
		data = usv[1].getArray();
		
		list.add(new LinkedList<String>());
		list.add(new LinkedList<String>());
		list.get(m).add("sigma");
		list.get(m).add(d0 + "x" + d1);
		
		for(int i = 0; i < d0; i++) {
			list.add(new LinkedList<String>());
			for(int j = 0; j < d1; j++) {
				list.get(i + m + 1).add(Double.toString(data[i][j]));
			}
		}
		
		for(int i = 0; i < list.size(); i++) {
			while (list.get(i).size() <= n) list.get(i).add("");
		}
		
		n += d1;
		
		d0 = usv[2].getSize()[0];
		d1 = usv[2].getSize()[1];
		data = usv[2].getArray();
		
		list.get(1).add("v");
		list.get(1).add(d0 + "x" + d1);
		
		for(int i = 0; i < d0; i++) {
			if (list.size() == d0) list.add(new LinkedList<String>());
			
			for(int j = 0; j < d1; j++) {
				list.get(i + 2).add(Double.toString(data[i][j]));
			}
		}
		
		for(int i = 0; i < list.size(); i++) {
			while (list.get(i).size() <= n) list.get(i).add("");
		}
		
		return list;
	}
	
	
	// */

	@Override
	public Block[] LUdecompostion() {
		throw new ArithmeticException("lu decomposing a low rank block");
	}
	
	public static LowRankBlock zeroBlock(int r, int c) {
		return new LowRankBlock(new Matrix(r, 0), new Matrix(0, 0), new Matrix(c, 0));
	}

}
