#ifndef _DEV_LOW_RANK_CUH
#define _DEV_LOW_RANK_CUH

#include <pspl.cuh>

template <class T> class dev_low_rank 
{
private:
  int nx;
  int ny;
  int rank;
  dev_dense <T> * U;
  dev_dense <T> * S;
  dev_dense <T> * VT;

public:

  __host__ dev_low_rank (const int x, const int y, const int rank_in = 0)
  {
    nx = x;
    ny = y;
    rank = (rank_in > 0 && (rank_in <= nx || rank_in <= ny)) ? rank_in : ((x > y) ? y : x);

    U = new dev_dense <T> (rank, ny);
    S = new dev_dense <T> (rank, rank);
    VT = new dev_dense <T> (rank, nx);

    S -> loadIdentityMatrix();
    VT -> loadIdentityMatrix();

  }

  __host__ ~dev_low_rank ()
  {
    delete U, S, VT;
  }

  __host__ int getRank () const
  {
    return rank;
  }

  __host__ dev_dense <T> * getU () const
  {
    return U;
  }

  __host__ dev_dense <T> * getS () const
  {
    return S;
  }

  __host__ dev_dense <T> * getVT () const
  {
    return VT;
  }

  __host__ T * getElementsU (const int offset = 0) const
  {
    return U -> getElements(offset);
  }

  __host__ T * getElementsS (const int offset = 0) const
  {
    return S -> getElements(offset);
  }

  __host__ T * getElementsVT (const int offset = 0) const
  {
    return VT -> getElements(offset);
  }

  __host__ void print() const
  {
    printf("\n-- Low Rank: %d x %d, rank %d --\n", nx, ny, rank);
    U -> print();
    S -> print();
    VT -> print();
  }

  __host__ dev_dense <T> * convertToDense() const
  {
    dev_dense<T> * t0 = U -> matrixMultiplication(S);
    dev_dense<T> * t1 = VT -> transpose();
    dev_dense<T> * t2 = t0 -> matrixMultiplication(t1);
    delete t0, t1;
    return t2;
  }

};


#endif