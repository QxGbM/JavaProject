

#include <stdio.h>
#include <dense.h>
#include <lowrank.h>

int main() {
  Dense d = Dense(4, 4);
  LowRank lr = LowRank(4, 4, 2);
  double* a = d.copyToCudaArray();
}

