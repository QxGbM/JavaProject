#pragma once

class matrix_painter
{

private:

  int nx;
  int ny;

  std :: vector <int> row;
  std :: vector <int> col;
  std :: vector <int> entry;

  void update_entry (const int entry_in, const int y, const int x);

  void clear_entries (const int y, const int x, const int ny = 1, const int nx = 1);

public:

  matrix_painter (const int nx_in, const int ny_in);

  ~matrix_painter ();

  int lookup_one (const int y, const int x) const;

  std::vector <int> * lookup (const int y, const int x, const int ny = 1, const int nx = 1) const;

  void update (const int entry, const int y, const int x, const int ny = 1, const int nx = 1);

  void print_internal () const;

  void print () const;

};

