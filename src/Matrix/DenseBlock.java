package Matrix;

import java.util.LinkedList;

public class DenseBlock extends Matrix implements Block{

	public DenseBlock(double[][] a) {
		super(a);
	}
	
	public DenseBlock(int r, int c) {
		super(r, c);
	}
	
	@Override
	public int[] getDimension() {
		return super.getSize();
	}

	@Override
	public boolean isLowRank() {
		return false;
	}
	
	@Override
	public boolean isHierarchical() {
		return false;
	}
	
	@Override
	public Matrix toMatrix() {
		return new Matrix(super.getArray());
	}
	
	@Override
	public DenseBlock toDense() {
		return this;
	}

	@Override
	public LowRankBlock toLowRank() {
		int[] size = super.getSize();
		return new LowRankBlock(toMatrix(), Math.max(size[0], size[1]));
	}

	@Override
	public Hierarchical toHierarchical() {
		int[] size = super.getSize();
		int m = size[0] / 2, n = size[1] / 2;
		Block e00 = new DenseBlock(super.block(0, m - 1 , 0, n - 1).getArray());
		Block e01 = new DenseBlock(super.block(0, m - 1, n, size[1] - 1).getArray());
		Block e10 = new DenseBlock(super.block(m, size[0] - 1, 0, n - 1).getArray());
		Block e11 = new DenseBlock(super.block(m, size[0] - 1, n, size[1] - 1).getArray());
		return new Hierarchical(e00, e01, e10, e11);
	}
	
	@Override
	public DenseBlock inverse() {
		return new DenseBlock(super.inverse().getArrayCopy());
	}

	@Override
	public Matrix[] usv() {
		return rsvd(Math.max(super.getSize()[0], super.getSize()[1]));
	}
	
	@Override
	public LinkedList<LinkedList<String>> blockString() {
		int d0 = super.getSize()[0], d1 = super.getSize()[1];
		double[][] data = super.getArray();
		LinkedList<LinkedList<String>> list = new LinkedList<LinkedList<String>>();
		list.add(new LinkedList<String>());
		list.getFirst().add("DenseBlock");
		list.getFirst().add(d0 + "x" + d1);
		
		while(list.getFirst().size() < d1) {
			list.getFirst().add("");
		}
		
		for(int i = 0; i < d0; i++) {
			list.add(new LinkedList<String>());
			for(int j = 0; j < d1; j++) {
				list.get(i + 1).add(Double.toString(data[i][j]));
			}
		}
		return list;
	}

	@Override
	public Block[] LUdecompostion() {
		Matrix[] lu = super.LUdecomposition();
		return new Block[] {new DenseBlock(lu[0].getArray()), new DenseBlock(lu[1].getArray())};
	}

	

}
