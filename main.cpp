

#include <ops.h>
#include <iostream>
#include <algorithm>
#include <random>

void test(int a, int ma, int na, int lda, int b, int mb, int nb, int ldb) {
  DataMap2D da(reinterpret_cast<void*>(a), lda, ma, na), db(reinterpret_cast<void*>(b), ldb, mb, nb);

  bool ol = false;
  for (int i = std::max(a, b); i < std::max(a + lda * na, b + ldb * nb); i++) {
    int xa = (i - a) / lda, ya = i - a - xa * lda;
    int xb = (i - b) / ldb, yb = i - b - xb * ldb;

    bool ina = xa < na && ya < ma;
    bool inb = xb < nb && yb < mb;

    ol |= ina && inb;
  }

  bool ol_t = da.checkOverlap(db);
  if (ol == ol_t)
    printf("P");
  else
    printf("FAIL %d %d x %d by %d, %d %d x %d by %d. %d %d\n", a, ma, na, lda, b, mb, nb, ldb, ol, ol_t);
}


int main(int argc, const char* argv[]) {

  for (int i = 0; i < 500; i++) {
    int a = std::rand() % 200, b = std::rand() % 200;
    int ma = 1 + std::rand() % 16, na = 1 + std::rand() % 16, oa = std::rand() % 16;
    int mb = 1 + std::rand() % 16, nb = 1 + std::rand() % 16, ob = std::rand() % 16;
    test(a, ma, na, ma + oa, b, mb, nb, mb + ob);
  }

  return 0;
}

