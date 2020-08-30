
#pragma once
#ifndef _Compress
#define _Compress

#include <matrix/Hierarchical.cuh>

class compressor {
private:
  vector<Dense*> d_lis;
  vector<LowRank*> lr_lis;

  void load(Hierarchical& h, const int rank, const double condition);

  void compress();

public:
  compressor(Hierarchical& h, const int rank, const double condition);

  ~compressor();


};

#endif