
public interface Block {

	enum Block_t { DENSE, LOW_RANK, HIERARCHICAL }
	
	abstract public int getRowDimension();

	abstract public int getColumnDimension();

	abstract public Block_t getType();
			
	abstract public Dense toDense();
	
	abstract public LowRank toLowRank();
	
	abstract public Hierarchical toHierarchical (int m, int n);

	abstract public String toString();

	abstract public boolean equals (Block b);

}
