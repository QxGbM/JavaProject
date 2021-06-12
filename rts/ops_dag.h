
#pragma once

class h_ops_dag 
{
private:

  int length;
  long long int * flops;
  long long int * flops_trim;
  h_ops_tree * ops_list;
  dependency_linked_list ** deps_graph_from;
  dependency_linked_list ** deps_graph_to;

  void build(const int nx, const int ny);

public:

  h_ops_dag (const h_ops_tree * ops, const int start_index = 0, const int length_max = 0);

  ~h_ops_dag ();

  int getLength () const;

  h_ops * getOp (const int index) const;

  dependency_t getDep (const int from, const int to) const;

  dependency_linked_list * getDepList_From (const int from) const;

  dependency_linked_list * getDepList_To (const int to) const;

  int getDepCount_From (const int from) const;

  int * getDepCountList_To () const;

  long long int getFlops (const int index = -1) const;

  long long int getFlops_Trim (const int index = -1) const;

  void print() const;

  double dag_density() const;
  
};


