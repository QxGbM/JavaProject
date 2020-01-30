
#include <definitions.cuh>
#include <h_ops/dependency_checker.cuh>


matrix_painter::matrix_painter (const int nx_in, const int ny_in)
{
  nx = nx_in > 0 ? nx_in : 1; 
  ny = ny_in > 0 ? ny_in : 1;

  row = std :: vector <int> (ny, 1); row[0] = 0;
  col = std :: vector <int> (1, 0);
  entry = std :: vector <int> (1, -1);
}

matrix_painter::~matrix_painter ()
{ }

void matrix_painter::update_entry (const int entry_in, const int y, const int x)
{
  if (y >= this -> ny || x >= this -> nx || y < 0 || x < 0) 
  { std :: cout << "Invalid Input at Matrix-Painter Update." << std ::endl; return; }

  auto * data_row = row.data();
  auto iter_col = col.begin() + data_row[y], iter_entry = entry.begin() + data_row[y];
  auto entry_end = (y >= ny - 1) ? entry.end() : entry.begin() + data_row[y + 1];

  while (* iter_col < x && iter_entry != entry_end)
  { iter_col++; iter_entry++; }

  if (iter_col == col.end())
  { 
    col.push_back(x); 
    entry.push_back(entry_in);
    for (int i = y + 1; i < ny; i++)
    { data_row[i]++; }
  }
  else if (* iter_col == x)
  { * iter_entry = entry_in; }
  else
  { 
    col.insert(iter_col, x);
    entry.insert(iter_entry, entry_in);
    for (int i = y + 1; i < ny; i++)
    { data_row[i]++; }
  }

}

void matrix_painter::clear_entries (const int y, const int x, const int ny, const int nx)
{
  if (y >= this -> ny || x >= this -> nx || y < 0 || x < 0) 
  { std :: cout << "Invalid Input at Matrix-Painter Clear Entry." << std ::endl; return; }

  auto * data_row = row.data();

  for (int i = y; i < y + ny && i < this -> ny; i++)
  {
    int n = (i == this -> ny - 1) ? ((int) entry.size() - data_row[i]) : (data_row[i + 1] - data_row[i]), count = n;
    if (n > 0)
    {
      auto start_col = col.begin() + data_row[i], start_entry = entry.begin() + data_row[i];
      auto end_col = start_col + n, end_entry = start_entry + n;

      while (* start_col < x)
      { start_col++; start_entry++; count--; }

      while (* (std :: prev(end_col)) >= x + nx)
      { end_col--; end_entry--; count--; }

      if (count > 0)
      {
        col.erase(start_col, end_col);
        entry.erase(start_entry, end_entry);
        for (int j = i + 1; j < this -> ny; j++)
        { data_row[j] -= count; }
      }
    }
  }
}

int matrix_painter::lookup_one (const int y, const int x) const
{
  if (y >= ny || x >= nx || y < 0 || x < 0) 
  { return -1; }
  else if (entry.size() == 0)
  { return -1; }

  auto * data_row = row.data();
  auto iter_col = col.begin(), iter_entry = entry.begin();
  auto entry_end = (y >= ny - 1) ? entry.end() : entry.begin() + data_row[y + 1];
  int ret = -1;

  while (iter_entry != entry_end) 
  {
    if (* iter_col <= x)
    { ret = * iter_entry; }
    iter_col ++; iter_entry ++; 
  }

  return ret;
}

std :: vector <int> * matrix_painter::lookup (const int y, const int x, const int ny, const int nx) const
{
  if (y >= this -> ny || x >= this -> nx || y < 0 || x < 0) 
  { return nullptr; }
  else if (entry.size() == 0)
  { return nullptr; }

  std :: unordered_set <int> set;
  std :: vector <int> tmp = std :: vector <int> (nx, lookup_one(y, x));

  auto data_row = row.data();

  for (int i = y; i < y + ny && i < this -> ny; i++)
  {
    int n = (i == this -> ny - 1) ? ((int) entry.size() - data_row[i]) : (data_row[i + 1] - data_row[i]), count = 1;
    auto iter_col = col.begin() + data_row[i], iter_entry = entry.begin() + data_row[i];

    if (n > 0)
    for (int j = 0; j < nx && j < (this -> nx - x); j++)
    {
      while (count < n && * (std :: next(iter_col)) <= j + x)
      { iter_col++; iter_entry++; count++; }

      if (j + x >= * iter_col)
      { tmp[j] = * iter_entry; }
    }

    for (int j : tmp)
    { if (j >= 0) { set.insert(j); } }
  }

  return set.size() > 0  ? new std :: vector <int> (set.begin(), set.end()) : nullptr; 

}


void matrix_painter::update (const int entry, const int y, const int x, const int ny, const int nx)
{
  if (y >= this -> ny || x >= this -> nx || y < 0 || x < 0) 
  { return; }

  std::vector <int> inst_y = std::vector <int> (ny, -1);
  std::vector <int> inst_x = std::vector <int> (nx, -1);

  if (x + nx < this -> nx) for (int i = 0; i < ny; i++)
  { inst_y[i] = lookup_one(y + i, x + nx); }

  if (y + ny < this -> ny) for (int i = 0; i < nx; i++)
  { inst_x[i] = lookup_one(y + ny, x + i); }

  if (ny > 1 || nx > 1)
  { clear_entries(y, x, ny, nx); }
  update_entry(entry, y, x);

  if (x + nx < this -> nx) for (int i = 0; i < ny; i++)
  { 
    if (lookup_one(y + i, x + nx) != inst_y[i])
    { update_entry(inst_y[i], y + i, x + nx); }
  }

  if (y + ny < this -> ny) for (int i = 0; i < nx; i++)
  {
    if (lookup_one(y + ny, x + i) != inst_x[i])
    { update_entry(inst_x[i], y + ny, x + i); }
  }
}

void matrix_painter::print_internal () const
{
  auto iter = row.begin();
  printf("Row: ");
  for (int i = 0; i < row.size(); i++)
  { printf("%d ", * iter); iter++; }
  
  iter = col.begin();
  printf("\nCol: ");
  for (int i = 0; i < col.size(); i++)
  { printf("%d ", * iter); iter++; }

  iter = entry.begin();
  printf("\nEntry: ");
  for (int i = 0; i < entry.size(); i++)
  { printf("%d ", * iter); iter++; }

  printf("\n");
}


void matrix_painter::print () const
{
  std :: vector <int> inst = std :: vector <int> (nx, -1);

  auto data_row = row.data();

  for (int y = 0; y < ny; y++)
  {
    int n = (y == ny - 1) ? ((int) entry.size() - data_row[y]) : (data_row[y + 1] - data_row[y]), count = 1;
    auto iter_col = col.begin() + data_row[y], iter_entry = entry.begin() + data_row[y];

    if (n > 0)
    for (int x = 0; x < nx; x++)
    {
      if (count < n && * (std :: next(iter_col)) <= x)
      { iter_col++; iter_entry++; count++; }

      if (x >= * iter_col)
      { inst[x] = * iter_entry; }
    }

    for (int x = 0; x < nx; x++)
    { printf("%d ", inst[x]); }

    printf("\n");
  }

}

/*int main()
{
  matrix_painter test = matrix_painter(4, 4);
  test.update(1, 0, 0);
  test.update(2, 0, 1, 1, 3);
  test.update(4, 1, 1, 3, 3);
  test.update(3, 1, 0, 3, 2);
  test.update(5, 1, 2);
  test.update(6, 2, 1);
  test.update(7, 2, 2);
  test.update(8, 2, 3);
  test.update(9, 3, 2);
  test.update(10, 3, 3);

  test.print_internal();
  test.print();

  std::vector<int> * vec = test.lookup(0, 1, 2, 2);

  for (int i : * vec)
  { printf("%d ", i); }
  
  delete vec;

  return 0;
}*/