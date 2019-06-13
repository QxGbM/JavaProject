
public class LowRank implements Block {
		
	private Dense U;
	private Dense S;
	private Dense VT;

	@Override
	public int getRowDimension() {
		return U.getRowDimension();
	}

  @Override
	public int getColumnDimension() {
		return VT.getRowDimension();
	}

	@Override
	public Block_t getType() 
	{ return Block_t.LOW_RANK; }
	
	@Override
	public Dense toDense() {
		return null;
	}

	@Override
	public LowRank toLowRank() {
		return this;
	}

	@Override
	public Hierarchical toHierarchical (int m, int n) {
		return null;
	}

	@Override
	public boolean equals(Block b) {
		return false;
	}

}
