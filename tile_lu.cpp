#include "../any.h"
#include "../low_rank.h"
#include "../hierarchical.h"
#include "../functions.h"
#include "../batch.h"
#include "../print.h"
#include "../timer.h"

#include "semaphore.h"
#include "pthread.h"

#include <algorithm>
#include <cmath>

using namespace hicma;

const int N = 64;
const int Nb = 16;
const int Nc = N / Nb;

Hierarchical A(Nc, Nc);

sem_t fin_sem[Nc][Nc], gemm_sem[Nc][Nc][Nc];

struct param
{
  int type;
  int ic;
  int jc;
};

void *thread (void *arg) {
  struct param *p = (struct param *) arg;

  if (p->type == 1) {
    int ic = p->ic;

    sem_wait(&gemm_sem[ic][ic][ic]);
    A(ic, ic).getrf();
    sem_post(&fin_sem[ic][ic]);

    for (int jc = ic + 1; jc < Nc; jc++) {
      sem_wait(&gemm_sem[ic][jc][ic]);
      A(ic, jc).trsm(A(ic, ic), 'l');
      sem_post(&fin_sem[ic][jc]);
    }
  }
  else if (p->type == 2) {
    int ic = p->ic, jc = p->jc;
    sem_wait(&gemm_sem[jc][ic][ic]);
    sem_wait(&fin_sem[ic][ic]);
    sem_post(&fin_sem[ic][ic]);
    A(jc, ic).trsm(A(ic, ic), 'u');
    for (int kc = ic + 1; kc < Nc; kc++) {
      sem_wait(&gemm_sem[jc][kc][ic]);
      sem_wait(&fin_sem[ic][kc]);
      sem_post(&fin_sem[ic][kc]);
      A(jc, kc).gemm(A(jc, ic), A(ic, kc));
      sem_post(&gemm_sem[jc][kc][ic + 1]);
    }
  }
  
  pthread_exit(0);
  return 0;
}

void LUdecomposition() {
  int num_cores = 4;
  pthread_t tid[num_cores];
  pthread_attr_t attr;
  pthread_attr_init(&attr);
  struct param para[num_cores];

  for (int i = 0; i < Nc; i++) {
    for (int j = 0; j < Nc; j++) {
      sem_init(&fin_sem[i][j], 0, 0);
      sem_init(&gemm_sem[i][j][0], 0, 1);
      for(int k = 1; k < Nc; k++) {
        sem_init(&gemm_sem[i][j][k], 0, 0);
      }
    }
  }

  int t0 = 0, t1 = 0;
  for (int i = 0; i < Nc; i++) {
    for (int j = i; j < Nc; j++) {
      if (t1) {
        pthread_join(tid[t0], NULL);
      }
      para[t0].type = (i == j) ? 1 : 2;
      para[t0].ic = i;
      para[t0].jc = j;
      pthread_create(&tid[t0], &attr, thread, &para[t0]);
      t0++;
      if(t0 == num_cores) {t0 = 0; t1 = 1;}
    }
  }

  int n = t0;

  if (t1) {
    do {
      pthread_join(tid[t0], NULL);
      t0++;
      if(t0 == num_cores) {t0 = 0;}
    } while (t0 != n);
  }
  else {
    for (int i = 0; i < t0; i++) {
      pthread_join(tid[i], NULL);
    }
  }
}

int main (int argc, char** argv) {
  
  std::vector<double> randx(N);

  Hierarchical x(Nc);
  Hierarchical b(Nc);

  for (int i=0; i<N; i++) {
    randx[i] = drand48();
  }
  std::sort(randx.begin(), randx.end());

  print("Time");
  start("Init matrix");
  for (int ic=0; ic<Nc; ic++) {
    Dense xi(Nb);
    Dense bj(Nb);
    for (int ib=0; ib<Nb; ib++) {
      xi[ib] = randx[Nb*ic+ib];
      bj[ib] = 0;
    }
    x[ic] = xi;
    b[ic] = bj;
  }
  for (int ic=0; ic<Nc; ic++) {
    for (int jc=0; jc<Nc; jc++) {
      Dense Aij(laplace1d, randx, Nb, Nb, Nb*ic, Nb*jc);
      A(ic,jc) = Aij;
    }
  }
  b.gemm(A, x, 1, 1);
  gemm_batch();
  stop("Init matrix");

  start("Tile LU decomposition");

  LUdecomposition();
  
  stop("Tile LU decomposition");

  printTime("-DGETRF");
  printTime("-DTRSM");
  printTime("-DGEMM");

  start("Forward substitution");
  for (int ic=0; ic<Nc; ic++) {
    for (int jc=0; jc<ic; jc++) {
      b[ic].gemm(A(ic,jc),b[jc]);
    }
    b[ic].trsm(A(ic,ic),'l');
  }
  stop("Forward substitution");

  printTime("-DTRSM");
  printTime("-DGEMM");

  start("Backward substitution");
  for (int ic=Nc-1; ic>=0; ic--) {
    for (int jc=Nc-1; jc>ic; jc--) {
      b[ic].gemm(A(ic,jc),b[jc]);
    }
    b[ic].trsm(A(ic,ic),'u');
  }
  stop("Backward substitution");

  printTime("-DTRSM");
  printTime("-DGEMM");

  double diff = (Dense(x) - Dense(b)).norm();
  double norm = x.norm();
  print("Accuracy");
  print("Rel. L2 Error", std::sqrt(diff/norm), false);
  return 0;
}
