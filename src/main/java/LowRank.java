
import Jama.Matrix;

public class LowRank implements Block {
		
private Matrix U;
private Matrix S;
private Matrix VT;

  public LowRank() {}

  @Override
  public int getRowDimension() 
  { return U.getRowDimension(); }

  @Override
  public int getColumnDimension() 
  { return VT.getRowDimension(); }

  @Override
  public Block_t getType() 
  { return Block_t.LOW_RANK; }

  @Override
  public Dense toDense() 
  {
    Dense d = new Dense(getRowDimension(), getColumnDimension());
    d.plusEquals(U.times(S).times(VT.transpose()));
    return d;
  }

  @Override
  public LowRank toLowRank() 
  { return this; }

  @Override
  public Hierarchical toHierarchical (int m, int n) {
  return null;
  }

  @Override
  public boolean equals (Block b) 
  {
    double norm = this.toDense().minus(b.toDense()).normF() / getRowDimension() / getColumnDimension();
    return norm < 1.e-10;
  }

  public void setU (Matrix U)
  { this.U = U; }

  public void setS (Matrix S)
  { this.S = S; }

  public void setVT (Matrix VT)
  { this.VT = VT; }

  public void print (int w, int d)
  { U.print(w, d); S.print(w, d); VT.print(w, d); }

}
