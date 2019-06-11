package Matrix;

import java.util.LinkedList;

public class Hierarchical implements Block{
	
	private int[] r = new int[2];
	private int[] c = new int[2];
	
	private Block[][] e = new Block[2][2];
	
	public Hierarchical (Matrix a, int k) {
		
		int[] size = a.getSize();
		r[0] = size[0] / 2;
		r[1] = size[0] - r[0];
		c[0] = size[1] / 2;
		c[1] = size[1] - c[0];
		
		e[0][0] = new LowRankBlock(a.block(0, r[0]-1, 0, c[0]-1), Math.min(r[0], c[0]));
		e[0][1] = new LowRankBlock(a.block(0, r[0]-1, c[0], c[0] + c[1]-1), Math.min(r[0], c[1]));
		e[1][0] = new LowRankBlock(a.block(r[0], r[0] + r[1]-1, 0, c[0]-1), Math.min(r[1], c[0]));
		e[1][1] = new LowRankBlock(a.block(r[0], r[0] + r[1]-1, c[0], c[0] + c[1]-1), Math.min(r[1], c[1]));
		
		for (int i = 0; i <= 1; i++) {
			for (int j = 0; j <= 1; j++) {
				Matrix[] usv = e[i][j].usv();
				int rank = Math.min(usv[1].getSize()[0], usv[1].getSize()[1]);
				if(rank > k) {
					e[i][j] = new Hierarchical(e[i][j].toMatrix(), k);
				}
				else if (rank == Math.min(r[i], c[j])){
					e[i][j] = new DenseBlock(e[i][j].toMatrix().getArray());
				}
			}
		}
	}
	
	public Hierarchical (Block a00, Block a01, Block a10, Block a11) {
		if (a00.getDimension()[0] != a01.getDimension()[0] ||
				a00.getDimension()[1] != a10.getDimension()[1]||
				a10.getDimension()[0] != a11.getDimension()[0]||
				a01.getDimension()[1] != a11.getDimension()[1])
			throw new ArithmeticException("Unmatched block dimension");
		else {
			r[0] = a00.getDimension()[0];
			r[1] = a10.getDimension()[0];
			c[0] = a00.getDimension()[1];
			c[1] = a01.getDimension()[1];
			e[0][0] = a00;
			e[0][1] = a01;
			e[1][0] = a10;
			e[1][1] = a11;
		}
			
	}
	
	@Override
	public int[] getDimension() {
		return new int[] {r[0] + r[1], c[0] + c[1]};
	}

	@Override
	public boolean isLowRank() {
		return false;
	}
	
	@Override
	public boolean isHierarchical() {
		return true;
	}
	
	@Override
	public Matrix toMatrix() {
		Matrix a = new Matrix(r[0] + r[1], c[0] + c[1]);
		double[][] data = a.getArray();
		
		double[][] m = e[0][0].toMatrix().getArray();
		for (int i = 0; i < r[0]; i++) {
			for (int j = 0; j < c[0]; j++) {
				data[i][j] = m[i][j];
			}
		}
		
		m = e[0][1].toMatrix().getArray();
		for (int i = 0; i < r[0]; i++) {
			for (int j = c[0]; j < c[0] + c[1]; j++) {
				data[i][j] = m[i][j - c[0]];
			}
		}
		
		m = e[1][1].toMatrix().getArray();
		for (int i = r[0]; i < r[0] + r[1]; i++) {
			for (int j = c[0]; j < c[0] + c[1]; j++) {
				data[i][j] = m[i - r[0]][j - c[0]];
			}
		}
		
		m = e[1][0].toMatrix().getArray();
		for (int i = r[0]; i < r[0] + r[1]; i++) {
			for (int j = 0; j < c[0]; j++) {
				data[i][j] = m[i - r[0]][j];
			}
		}
		
		return a;
	}
	
	@Override
	public DenseBlock toDense() {
		return new DenseBlock(toMatrix().getArray());
	}

	@Override
	public LowRankBlock toLowRank() {
		return toDense().toLowRank();
	}

	@Override
	public Hierarchical toHierarchical() {
		return this;
	}

	@Override
	public Matrix[] usv() {
		return null;
	}

	@Override
	public LinkedList<LinkedList<String>> blockString() {
		LinkedList<LinkedList<String>> list = new LinkedList<LinkedList<String>>();
		
		LinkedList<LinkedList<String>> l0 = e[0][0].blockString();
		LinkedList<LinkedList<String>> l1 = e[0][1].blockString();
		LinkedList<LinkedList<String>> l2 = e[1][0].blockString();
		LinkedList<LinkedList<String>> l3 = e[1][1].blockString();
		
		list.addAll(l0);
		do {
			LinkedList<String> l = new LinkedList<String>();
			while(l.size() < list.getFirst().size()) l.add("");
			list.add(l);
		} while (list.size() <= l1.size());
		
		int n = list.size(), o = l2.getFirst().size(), p = l0.getFirst().size();
		
		for (int i = 0; i < l1.size(); i++) {
			LinkedList<String> l = list.get(i);
			do {
				l.add("");
			} while (l.size() <= o);
			l.addAll(l1.get(i));
		}
		
		list.addAll(l2);
		int size = list.get(n).size();
		
		while (list.size() < n + l3.size()){
			LinkedList<String> l = new LinkedList<String>();
			while(l.size() < size) l.add("");
			list.add(l);
		}
		
		for (int i = 0; i < l3.size(); i++) {
			LinkedList<String> l = list.get(n + i);
			do {
				l.add("");
			} while (l.size() <= p);
			l.addAll(l3.get(i));
		}
		
		int max = 0;
		for (int i = 0; i < list.size(); i++) {
			if (list.get(i).size() > max) max = list.get(i).size();
		}
		
		for (int i = 0; i < list.size(); i++) {
			while(list.get(i).size() < max) list.get(i).add("");
		}
		return list;
	}

	@Override
	public Block[] LUdecompostion() {
		Block[] a = e[0][0].LUdecompostion();
		Block l00 = a[0], u00 = a[1];
		Block l10 = Block.multiply(e[1][0], u00.inverse());
		Block u01 = Block.multiply(l00.inverse(), e[0][1]);
		
		Block[] b = Block.subtract(e[1][1], Block.multiply(l10, u01)).LUdecompostion();
		Block l11 = b[0], u11 = b[1];
		
		Block l01 = LowRankBlock.zeroBlock(l00.getDimension()[0], l11.getDimension()[1]);
		Block u10 = LowRankBlock.zeroBlock(u11.getDimension()[0], u00.getDimension()[1]);
		
		Block l = new Hierarchical(l00, l01, l10, l11);
		Block u = new Hierarchical(u00, u01, u10, u11);
		return new Block[] {l, u};
	}
	
	public Block[][] getElements() {
		return e;
	}
}
