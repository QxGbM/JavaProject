#ifndef _DEV_LOW_RANK_CUH
#define _DEV_LOW_RANK_CUH

#include <pspl.cuh>

template <class T> class dev_low_rank 
{
private:
  int nx;
  int ny;
  int rank;
  dev_dense <T> * UxS;
  dev_dense <T> * VT;

public:

  __host__ dev_low_rank (const int x, const int y)
  {
    nx = x;
    ny = y;
    rank = (x > y) ? y : x;

    UxS = new dev_dense <T> (rank, ny);
    VT = new dev_dense <T> (rank, nx);

    VT -> loadIdentityMatrix();

  }

  __host__ ~dev_low_rank ()
  {
    delete UxS;
    delete VT;
  }

  __host__ inline int getNx () const { return nx; }

  __host__ inline int getNy () const { return ny; }

  __host__ inline int getRank () const { return rank; }

  __host__ inline dev_dense <T> * getUxS () const { return UxS; }

  __host__ inline dev_dense <T> * getVT () const { return VT; }

  __host__ inline int getNx_UxS() const { return UxS -> getNx(); }

  __host__ inline int getNy_UxS() const { return UxS -> getNy(); }

  __host__ inline int getLd_UxS() const { return UxS -> getLd(); }

  __host__ inline int getOffset_VT() const { return getNy_UxS() * getLd_UxS(); }

  __host__ inline int getNx_VT() const { return VT -> getNx(); }

  __host__ inline int getNy_VT() const { return VT -> getNy(); }

  __host__ inline int getLd_VT() const { return VT -> getLd(); }

  __host__ inline T * getElements (const int offset = 0) const 
  { 
    return offset >= getOffset_VT() ? VT -> getElements (offset - getOffset_VT()) : UxS -> getElements(offset); 
  }

  __host__ T getElement (const int x, const int y) const
  {
    T element = 0, * row = UxS -> getElements (y * rank), * col = VT -> getElements (x * rank);
    for (int i = 0; i < rank; i++)
    { element += row[i] * col[i]; }
    return element;
  }

  __host__ void adjustRank (const int rank_in)
  {
    if (rank_in > 0 && rank_in != rank)
    {
      UxS -> resize (rank_in, ny);
      VT -> resize (rank_in, nx);
      rank = rank_in;
    }
  }

  __host__ dev_dense <T> * convertToDense() const
  {
    dev_dense<T> * t1 = VT -> transpose();
    dev_dense<T> * t2 = UxS -> matrixMultiplication(t1);
    delete t1;
    return t2;
  }


  __host__ void print() const
  {
    printf("\n-- Low Rank: %d x %d, rank %d --\n", nx, ny, rank);
    UxS -> print();
    VT -> print();
  }

  __host__ void loadTestMatrix(const int x_start = 0, const int y_start = 0) const
  {
    UxS -> loadTestMatrix (x_start, y_start);
  }

};


#endif