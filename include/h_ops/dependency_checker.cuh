#pragma once
#ifndef _DEP_CHECK_CUH
#define _DEP_CHECK_CUH

#include <definitions.cuh>
#include <vector>

class matrix_painter
{

private:

  int nx;
  int ny;

  std :: vector <int> row;
  std :: vector <int> col;
  std :: vector <int> entry;

  void update_entry (const int entry_in, const int y, const int x);
  
public:

  matrix_painter (const int nx_in, const int ny_in);

  ~matrix_painter ();

  int lookup_one (const int y, const int x) const;

  int * lookup (int * result_length_out, const int y, const int x, const int ny = 1, const int nx = 1) const;

  void update (const int entry, const int y, const int x, const int ny = 1, const int nx = 1);

  void print_internal () const;

  void print () const;

};

#endif