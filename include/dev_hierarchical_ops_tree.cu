
#include <pspl.cuh>

h_ops_tree::h_ops_tree () : h_ops ()
{ l_children = 0; children = nullptr; }

h_ops_tree::h_ops_tree (const operation_t op_in, const h_index * M) : h_ops (op_in, M)
{ l_children = 0; children = nullptr; }

h_ops_tree::h_ops_tree (const operation_t op_in, const h_index * M1, const h_index * M2) : h_ops (op_in, M1, M2)
{ l_children = 0; children = nullptr; }

h_ops_tree::h_ops_tree (const operation_t op_in, const h_index * M1, const h_index * M2, const h_index * M3) : h_ops (op_in, M1, M2, M3)
{ l_children = 0; children = nullptr; }

h_ops_tree::~h_ops_tree ()
{ 
  if (l_children > 0) 
  { delete[] children; } 
}

h_ops_tree * h_ops_tree::getChild (const int index) const
{ return (index >= l_children) ? nullptr : &children[index]; }

void h_ops_tree::setChild (h_ops_tree * op, const int index)
{
  if (index >= 0)
  { 
    if (index >= l_children) 
    { resizeChildren(index + 1); } 
    op -> clone(&children[index], true); 
  }
  else if (index == -1)
  { 
    resizeChildren(l_children + 1); 
    op -> clone(&children[l_children - 1], true); 
  }

}

void h_ops_tree::resizeChildren (const int length_in)
{
  if (length_in > 0 && length_in != l_children)
  { 
    h_ops_tree * neo = new h_ops_tree [length_in];

    for (int i = 0; i < l_children && i < length_in; i++)
    { children[i].clone(&neo[i], true); }

    if (l_children > 0) 
    { delete[] children; }

    children = neo;
    l_children = length_in;
  }
}

int h_ops_tree::length () const
{
  if (this == nullptr)
  { return 0; }
  else if (l_children == 0) 
  { return 1; }
  else
  {
    int length_ = 0;
#pragma omp parallel for reduction (+:length_) if (omp_in_parallel() == 0)
    for (int i = 0; i < l_children; i++) 
    { length_ += children[i].length(); }
    return length_;
  }
}

h_ops_tree * h_ops_tree::clone (h_ops_tree * addr, const bool clone_child) const
{
  if (this == nullptr)
  { return nullptr; }
  else if (addr == nullptr)
  { h_ops_tree * op = new h_ops_tree(); clone(op); return op; }
  else
  {
    if (opType() == gemm)
    { 
      addr -> op_type = op_type;
      addr -> read_and_write = new h_index[1];
      read_and_write[0].clone(&(addr -> read_and_write)[0]);
      addr -> n_rw = 1;

      addr -> read_only = new h_index[2];
      read_only[0].clone(&(addr -> read_only)[0]);
      read_only[1].clone(&(addr -> read_only)[1]);
      addr -> n_ro = 2;
    }
    else if (opType() >= trsml && opType() <= pivot)
    {         
      addr -> op_type = op_type;
      addr -> read_and_write = new h_index[1];
      read_and_write[0].clone(&(addr -> read_and_write)[0]);
      addr -> n_rw = 1;

      addr -> read_only = new h_index[1];
      read_only[0].clone(&(addr -> read_only)[0]);
      addr -> n_ro = 1;
    }
    else if (opType() == getrf)
    {         
      addr -> op_type = op_type;
      addr -> read_and_write = new h_index[1];
      read_and_write[0].clone(&(addr -> read_and_write)[0]);
      addr -> n_rw = 1;

      addr -> read_only = nullptr;
      addr -> n_ro = 0;
    }
    else
    {
      addr -> op_type = op_type;
      addr -> read_and_write = nullptr;
      addr -> n_rw = 0;
      addr -> read_only = nullptr;
      addr -> n_ro = 0;
    }

    if (clone_child)
    {
      addr -> l_children = l_children;
      addr -> children = (l_children > 0) ? new h_ops_tree [l_children] : nullptr;
      for (int i = 0; i < l_children; i++)
      { children[i].clone(&(addr -> children)[i], clone_child); }
    }
    else
    {
      addr -> l_children = 0;
      addr -> children = nullptr;
    }
      
    addr -> flops = flops;
    return addr;
  }
}

h_ops_tree * h_ops_tree::flatten (const int start_index, const int length_max, const int list_index, h_ops_tree * list) const
{

  int length_ = 0, * lengths = new int [l_children];

#pragma omp parallel for reduction (+:length_) if (omp_in_parallel() == 0)
  for (int i = 0; i < l_children; i++) 
  { const int l = children[i].length(); lengths[i] = l; length_ += l; }

  if (length_ <= start_index)
  { delete[] lengths; return nullptr; }
  else
  { length_ = (length_max > 0 && length_max <= length_ - start_index) ? length_max : length_ - start_index; }

  if (list == nullptr)
  {
    list = clone (nullptr, false);
    list -> resizeChildren (length_);
  }

  int child_start = 0, child_end = l_children, insts_read = 0, insts_start = 0, end_length = 0;
  for (int i = 0; i < l_children; i++)
  {
    if (insts_read <= start_index)
    { child_start = i; insts_start = start_index - insts_read; }

    end_length = start_index + length_ - insts_read;

    if (end_length <= lengths[i])
    { child_end = i + 1; end_length = end_length > length_max ? length_max : end_length; break; }
    else
    { insts_read += lengths[i]; }
  }

  int iters = child_end - child_start;
  if (iters > 1)
  {
    int * work_index = new int [iters];
    work_index[0] = list_index;
    work_index[1] = list_index + lengths[child_start] - insts_start;

    for (int i = 1; i < iters - 1; i++)
    { work_index[i + 1] = work_index[i] + lengths[i + child_start]; }

#pragma omp parallel for if (omp_in_parallel() == 0)
    for (int i = child_start; i < child_end; i++)
    {
      const int index = work_index[i - child_start];
      if (children[i].l_children == 0)
      { children[i].clone(&(list -> children)[index], false); }
      else if (i == child_start)
      { children[i].flatten (insts_start, 0, index, list); }
      else if (i == child_end - 1)
      { children[i].flatten (0, end_length, index, list); }
      else
      { children[i].flatten (0, 0, index, list); }
    }
    delete[] work_index;
  }
  else
  {
    if (children[child_start].l_children == 0)
    { children[child_start].clone(&(list -> children)[list_index], false); }
    else
    { children[child_start].flatten (insts_start, end_length, list_index, list); }
  }

  delete[] lengths;
  return list;
}

long long int h_ops_tree::getFlops(long long int * trim)
{
  if (this == nullptr)
  { return 0; }
  else if (l_children == 0)
  { return h_ops::getFlops(trim); }
  else
  { 
    long long int accum = 0, accum_trim = 0;

#pragma omp parallel for reduction (+:accum, accum_trim) if (omp_in_parallel == 0) 
    for (int i = 0; i < l_children; i++) 
    {
      long long int tmp;
      accum += children[i].getFlops(&tmp);
      accum_trim += tmp;
    }

    * trim = accum_trim;
    return accum;
  }
}

void h_ops_tree::print (const int op_id, const int indent) const
{
  for (int i = 0; i < indent; i++) { printf("  "); }

  if (l_children == 0) { printf("%d: ", op_id); }

  h_ops::print();

  int offset = 0, l = length();
  for (int i = 0; i < l_children && offset < l; offset += children[i].length(), i++)
  { children[i].print(op_id + offset, indent + 1); }
}

