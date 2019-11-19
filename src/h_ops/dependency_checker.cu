
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
  auto * data_row = row.data();
  auto iter_col = col.begin() + data_row[y], iter_entry = entry.begin() + data_row[y];

  while (* iter_col < x && iter_col != col.end())
  { iter_col++; iter_entry++; }

  if (* iter_col == x)
  { * iter_entry = entry_in; }
  else
  { 
    col.insert(iter_col, x); 
    entry.insert(iter_entry, entry_in); 
    for (int i = y + 1; i < ny; i++)
    { data_row[i]++; }
  }

}

int matrix_painter::lookup_one (const int y, const int x) const
{
  auto * data_row = row.data();
  auto iter_col = col.begin(), iter_entry = entry.begin();

  if (data_row[y] > 0)
  {
    for (int i = 0; i < y; i++)
    { iter_col += data_row[i]; iter_entry += data_row[i]; }

    while (* iter_col < x) 
    { iter_col ++; iter_entry ++; }

    return * iter_entry;
  }
  else
  {
    int ret = -1;
    for (int i = 0; i < y; i++)
    {
      for (int j = 0; j < data_row[i]; j++)
      {
        if (* iter_col <= x) { ret = * iter_entry; }
        iter_col ++; iter_entry ++;
      }
    }

    return ret;
  }
}


void matrix_painter::update (const int entry, const int y, const int x, const int ny, const int nx)
{
  update_entry(entry, y, x);
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
    int n = (y == ny - 1) ? entry.size() - data_row[y] : data_row[y + 1] - data_row[y], count = 1;
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

int main()
{
  matrix_painter test = matrix_painter(4, 4);
  test.update(1, 0, 0);
  test.update(2, 0, 1);
  test.update(4, 1, 1);
  test.update(3, 1, 0);
  test.update(5, 1, 2);
  test.update(6, 2, 1);
  test.update(7, 2, 2);
  test.update(8, 2, 3);
  test.update(9, 3, 2);
  test.update(10, 3, 3);
  test.print_internal();
  test.print();
  return 0;
}