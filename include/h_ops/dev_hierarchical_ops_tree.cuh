
#pragma once
#ifndef _DEV_HIERARCHICAL_OPS_TREE_CUH
#define _DEV_HIERARCHICAL_OPS_TREE_CUH

#include <definitions.cuh>
#include <h_ops/dev_hierarchical_ops.cuh>

class h_ops_tree : public h_ops
{
private:
  int l_children;
  h_ops_tree * children;

public:

  h_ops_tree ();

  h_ops_tree (const operation_t op_in, const h_index * M);

  h_ops_tree (const operation_t op_in, const h_index * M1, const h_index * M2);

  h_ops_tree (const operation_t op_in, const h_index * M1, const h_index * M2, const h_index * M3);

  ~h_ops_tree ();

  h_ops_tree * getChild (const int index) const;

  void setChild (h_ops_tree * op, const int index = -1);

  void resizeChildren (const int length_in);

  int length () const;

  h_ops_tree * clone (h_ops_tree * addr = nullptr, const bool clone_child = false) const;

  h_ops_tree * flatten (const int start_index = 0, const int length_max = 0, const int list_index = 0, h_ops_tree * list = nullptr) const;

  long long int getFlops(long long int * trim);

  void print (const int op_id = 0, const int indent = 0) const;

};

#endif