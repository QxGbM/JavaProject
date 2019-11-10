
#include <definitions.cuh>
#include <h_ops/dependency.cuh>

dependency_t operator+ (dependency_t dep1, dependency_t dep2)
{ return (dependency_t) ((int) dep1 | (int) dep2); }

bool operator> (dependency_t dep1, dependency_t dep2)
{ return (int) dep1 > (int) dep2; }

dependency_linked_list::dependency_linked_list (const int inst_in, const dependency_t dependency_in, dependency_linked_list * next_in)
{
  inst = inst_in;
  dependency = dependency_in;
  next = next_in;
}

dependency_linked_list::~dependency_linked_list ()
{ delete next; }

int dependency_linked_list::getInst () const
{ return inst; }

dependency_t dependency_linked_list::getDep () const
{ return dependency; }

dependency_linked_list * dependency_linked_list::getNext () const
{ return next; }

dependency_t dependency_linked_list::lookupDependency (const int inst_in) const
{
  for (const dependency_linked_list * ptr = this; ptr != nullptr; ptr = ptr -> next)
  { 
    if (ptr -> inst == inst_in) { return ptr -> dependency; }
    else if (ptr -> inst < inst_in) { return dependency_t::no_dep; }
  }
  return dependency_t::no_dep;
}

int dependency_linked_list::length () const
{
  int l = 0;
  for (const dependency_linked_list * ptr = this; ptr != nullptr; ptr = ptr -> next) { l++; }
  return l;
}

void dependency_linked_list::print () const
{
  for (const dependency_linked_list * ptr = this; ptr != nullptr; ptr = ptr -> next) { printf("%d ", ptr -> inst); }
  printf("\n");
}



dependency_table::dependency_table (int size_in)
{
  size_in = size_in >= 1 ? size_in : 1;
  size = size_in;
  from = new dependency_linked_list * [size_in];
  to = new dependency_linked_list * [size_in];

  memset(from, 0, sizeof(dependency_linked_list *) * size_in);
  memset(to, 0, sizeof(dependency_linked_list *) * size_in);
}

dependency_table::~dependency_table()
{
  for (int i = 0; i < size; i++)
  { delete from[i]; delete to[i]; }

  delete[] from;
  delete[] to;
}

void dependency_table::resize (int size_new)
{
  size_new = size_new >= 1 ? size_new : 1;

  dependency_linked_list ** from_new = new dependency_linked_list * [size_new];
  dependency_linked_list ** to_new = new dependency_linked_list * [size_new];

  for (int i = 0; i < size_new && i < size; i++)
  { from_new[i] = from[i]; to_new[i] = to[i]; }

  for (int i = size; i < size_new; i++)
  { delete from[i]; delete to[i]; from_new[i] = nullptr; to_new[i] = nullptr; }

  delete[] from; from = from_new;
  delete[] to; to = to_new;
}

void dependency_table::add_dep (const int inst_from, const int inst_to, dependency_t type)
{
  if (inst_from >= 0 && inst_to >= 0 && inst_from < size && inst_to < size && inst_from < inst_to)
  {

  }
}

