

#include <stdio.h>
#include <dense.h>

int main() {
  Dense d = Dense(4, 4);
  double* a = d.copyToCudaArray();
}

