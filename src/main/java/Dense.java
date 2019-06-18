
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
    int m = getRowDimension(), n = getColumnDimension();
    int step = n > 32 ? 8 : n / 4, r = 0;

    boolean approx;
    Matrix Q, Y;
    QRDecomposition qr_;
    
    do {
      r += step; r = r >= n ? n : r;

      Matrix X = times(random(n, r));
      qr_ = X.qr();
      Q = qr_.getQ();
      Y = transpose().times(Q);
      Matrix A = Q.times(Y.transpose()).minus(this);
      double norm = A.normF() / (m * r);
      approx = norm <= 1.e-10;

    } while (r < n && !approx);

    qr_ = Y.qr();
    Matrix V = qr_.getQ(), R = qr_.getR();
    SingularValueDecomposition svd_ = R.svd();

    LowRank lr = new LowRank (m, n, r);
    lr.setU(Q.times(svd_.getV()));
    lr.setS(svd_.getS());
    lr.setVT(V.times(svd_.getU()));

    return lr;
  }

  @Override
  public Hierarchical toHierarchical (int m, int n)
  {
    Hierarchical h = new Hierarchical(m, n);
    int i0 = 0;
    int step_i = getRowDimension() / m - 1, step_j = getColumnDimension() / n - 1;

    for (int i = 0; i < m; i++)
    {
      int i1 = i0 + step_i >= getRowDimension() ? getRowDimension() - 1 : i0 + step_i, j0 = 0;
      for (int j = 0; j < n; j++)
      {
        int j1 = j0 + step_j >= getColumnDimension() ? getColumnDimension() - 1 : j0 + step_j;
        Dense d = new Dense(i1 - i0 + 1, j1 - j0 + 1); 
        d.plusEquals(getMatrix(i0, i1, j0, j1));
        h.setElement(i, j, d);
        j0 = j1 + 1;
      }
      i0 = i1 + 1;
    }

    return h;
  }

  @Override
  public boolean equals (Block b) 
  {
    double norm = this.minus(b.toDense()).normF() / getColumnDimension() / getRowDimension();
    return norm <= 1.e-10; 
  }

  @Override
  public void print (int w, int d)
  { super.print(w, d); }


}
