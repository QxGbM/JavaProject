import java.io.*;

public interface Block {

  enum Block_t { DENSE, LOW_RANK, HIERARCHICAL }

  abstract public int getRowDimension();

  abstract public int getColumnDimension();

  abstract public Block_t getType();
		
  abstract public Dense toDense();

  abstract public LowRank toLowRank();

  abstract public Hierarchical toHierarchical (int m, int n);

  abstract public boolean equals (Block b);

  abstract public String structure ();

  abstract public void writeBinary (OutputStream stream) throws IOException;

  abstract public void writeToFile (String name) throws IOException;

  abstract public void print (int w, int d);

}
