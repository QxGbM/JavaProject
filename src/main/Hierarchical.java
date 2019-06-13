
public class Hierarchical implements Block{

	@Override
	public int getRowDimension() {
		return 0;
	}

	@Override
	public int getColumnDimension() {
		return 0;
	}

	@Override
	public Block_t getType() {
		return null;
	}

	@Override
	public Dense toDense() {
		return null;
	}

	@Override
	public LowRank toLowRank() {
		return null;
	}

	@Override
	public Hierarchical toHierarchical(int m, int n) {
		return null;
	}

	@Override
	public boolean equals(Block b) {
		return false;
	}
	
	
}
