
import Jama.Matrix;
import Jama.QRDecomposition;
import Jama.SingularValueDecomposition;

public class Dense extends Matrix implements Block 
{

  private static final long serialVersionUID = 1;

  public Dense (double[][] A)
  { super(A); }

  public Dense (double[][] A, int m, int n) 
  { super(A, m, n); }

  public Dense (double[] vals, int m) 
  { super(vals, m); }

  public Dense (int m, int n)
  { super(m, n); }

  public Dense (int m, int n, double s)
  { super(m, n, s); }

  @Override
  public int getRowDimension() 
  { return super.getRowDimension(); }

  @Override
  public int getColumnDimension() 
  { return super.getColumnDimension(); }

  @Override
  public Block_t getType()
  { return Block_t.DENSE; }

  @Override
  public Dense toDense()
  { return this; }

  @Override
  public LowRank toLowRank()
  {
    int step = getColumnDimension() > 32 ? 8 : getColumnDimension() / 4, r = step;
    QRDecomposition qr_;
    
    do {
      Matrix X = this.times(random(getColumnDimension(), r));
      qr_ = X.qr();
      r += step;
    } while (qr_.isFullRank());

    Matrix Q = qr_.getQ();
    Matrix Y = this.transpose().times(Q);
    qr_ = Y.qr();
    Matrix V = qr_.getQ();
    Matrix R = qr_.getR();
    SingularValueDecomposition svd_ = R.svd();

    LowRank lr = new LowRank();
    lr.setU(Q.times(svd_.getU()));
    lr.setS(svd_.getS());
    lr.setVT(V.times(svd_.getV().transpose()));

    return lr;
  }

  @Override
  public Hierarchical toHierarchical (int m, int n)
  {
    return null;
  }

  @Override
  public String toString() {
    return null;
  }

  @Override
  public boolean equals (Block b) 
  {
    double norm = this.minus(b.toDense()).normF() / getColumnDimension() / getRowDimension();
    return norm <= 1.e-10; 
  }


}
