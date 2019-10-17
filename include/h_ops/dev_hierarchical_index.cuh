
#pragma once
#ifndef _DEV_HIERARCHICAL_INDEX_CUH
#define _DEV_HIERARCHICAL_INDEX_CUH

#include <pspl.cuh>

class h_index 
{
private:

  int index_lvls;
  int * indexs;

  element_t type;
  int nx;
  int ny;
  int ld_x;
  int ld_y;
  int offset_x;
  int offset_y;
  int rank;

  int n_ptrs;
  void ** data_ptrs;
  int tmp_id;

  const void * root_ptr;

public:

  h_index();

  h_index(const h_index * index);

  h_index(const dev_hierarchical * h);

  h_index(const dev_hierarchical * h, const h_index * index, const int y, const int x);

  h_index(const h_index * index, const int y_start, const int x_start, const int ny_block, const int nx_block);

  h_index(const dev_dense * d);

  h_index(const dev_low_rank * lr);

  ~h_index();

  int getNx() const;

  int getNy() const;

  int getSize() const;

  int getNx(const int min) const;

  int getNy(const int min) const;

  int getLd_x() const;

  int getLd_y() const;

  int getOffset() const;

  int getOffset_x() const;

  int getOffset_y() const;

  int getRank() const;

  int getRank (const int min) const;

  int getTranspose() const;

  relation_t compare (const h_index * index) const;

  h_index * clone (h_index * addr = nullptr) const;

  void setU ();

  void setVT ();

  void setTemp_Dense (const int block_id);

  void setTemp_Low_Rank (const int block_id, const int rank_in);

  void setShadow (const h_index * parent);

  void setU_data (void * u_in, const int offset_y_in, const int ld_y_in);

  void setVT_data (void * vt_in, const int offset_x_in, const int ld_x_in);

  void setU_data (const h_index * index);

  void setVT_data (const h_index * index);

  int getMinRank (const h_index * index, bool * a = nullptr) const;

  int getDataPointers (void ** data_ptrs_in, void ** tmp_ptrs) const;

  bool isDense () const;

  bool isLowRank () const;

  bool isLowRank_Full () const;

  bool isU () const;

  bool isVT () const;

  void print() const;

};


#endif