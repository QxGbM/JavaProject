
import Jama.Matrix;

public class Dense extends Matrix implements Block {

	private int level;
	private int[] row_indices;
	private int[] column_indices;

	private static final long serialVersionUID = 1;
	
	public Dense (double[][] A)
	{ super(A);	}
	
	public Dense (double[][] A, int m, int n) 
	{ super(A, m, n); }
	
	public Dense (double[] vals, int m) 
	{ super(vals, m); }

	public Dense (int m, int n)
	{ super(m, n); }

	public Dense (int m, int n, double s)
	{ super(m, n, s); }

	@Override
	public int getRowDimension() {
		return super.getRowDimension();
	}

  @Override
	public int getColumnDimension() {
		return super.getColumnDimension();
	}

	@Override
	public Block_t getType()
	{ return Block_t.DENSE; }

	@Override
	public Dense toDense()
	{ return this; }
	
	@Override
	public LowRank toLowRank()
	{ 
		// TODO
		return null;
	}
	
	@Override
	public Hierarchical toHierarchical (int m, int n)
	{
		// TODO
		return null;
	}
	
	@Override
	public String toString() {
		// TODO
		return null;
	}
	
	@Override
	public boolean equals (Block b) {

		Jama.Matrix diff = this.minus(b.toDense());
		double norm = diff.normF() / getColumnDimension() / getRowDimension();
		if (norm <= 1.e-10) 
		{ return true; }
    else
		{ return false; }
	}
	
	
}
