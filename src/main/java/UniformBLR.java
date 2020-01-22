
import Jama.Matrix;
import Jama.SingularValueDecomposition;

public class UniformBLR {

  Matrix rows[];
  Matrix cols[];
  Block e[][];

  public UniformBLR (int m, int n) {
    e = new Block[m][n];
    rows = new Matrix[m];
    cols = new Matrix[n];    
  }

  public UniformBLR (int m, int n, Dense d, int rank) {
    Hierarchical h = d.toHierarchical(m, n);
    e = h.getElements();
    rows = new Matrix[m];
    cols = new Matrix[n];

    for (int i = 1; i < m; i++)
    {
      Dense blocks = e[i][0].toDense();
      Matrix[] usv = blocks.rsvd(rank);
      rows[i] = new Matrix(usv[0].getArray());
    }
    rows[0] = new Matrix(e[0][1].toDense().rsvd(rank)[0].getArray());

    for (int i = 1; i < n; i++)
    {
      Dense blocks = e[0][i].toDense().transpose();
      Matrix[] usv = blocks.rsvd(rank);
      cols[i] = new Matrix(usv[0].getArray());
    }
    cols[0] = new Matrix(e[1][0].toDense().transpose().rsvd(rank)[0].getArray());
    /*SingularValueDecomposition svd_ = e[0][0].toDense().svd(), svd_2 = e[0][0].toDense().transpose().svd();
    rows[0] = svd_.getU().getMatrix(0, 31, 0, 15);
    cols[0] = svd_2.getU().getMatrix(0, 31, 0, 15);*/

    for (int i = 0; i < m; i++) {
      System.out.print(i);
      for (int j = 0; j < n; j++) {
        boolean admis = e[i][j].testAdmis(rows[i], cols[j], 1e-7);
        System.out.print(" " + admis);
        if (admis) {
          e[i][j] = new LowRank(e[i][j].toDense(), rows[i], cols[j]);
        }

      }
      System.out.println();
    }

    Dense t = e[0][0].toDense();
    Matrix list[] = t.projection(rows[0], cols[0]);
    
    Matrix test1 = rows[0].times(list[0]).times(cols[0].transpose());
    test1.plusEquals(list[4].times(list[3]).times(list[5].transpose()));
    test1.minusEquals(t);
    System.out.println("diag err: " + test1.normF());

    test1.plusEquals(list[4].times(list[1]).times(cols[0].transpose()));
    test1.plusEquals(rows[0].times(list[2]).times(list[5].transpose()));
    System.out.println("all 4 err: " + test1.normF());


  }

  
}