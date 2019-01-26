
#include <gpu_lu.cuh>

int main(int argc, char **argv)
{
  const unsigned nx = 64;
  const unsigned ny = 32;

  //test_dense_getrf_1x1 ();
  //test_dense_getrf_nx1 (ny);
  //test_dense_getrf_1xn (nx);
  test_dense_getrf_nxn (nx, ny);

  //test_inverse (nx, ny);
  
  // Failed Test.
  //test_dense_gemm_1block(nx, ny); 

  return 0;
}