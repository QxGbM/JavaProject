

#include <ops.h>
#include <dependency.h>

dependency_t Operation::checkDependencyFrom (const Operation& op_from) const {
  int rw_from = op_from -> n_rw, ro_from = op_from -> n_ro, rw_to = n_rw, ro_to = n_ro, dep = (int) dependency_t::no_dep;

  for (int i = 0; i < rw_from * (rw_to + ro_to); i++)
  {
    const int to = i / rw_from, from = i - to * rw_from;

    if (to < rw_to)
    {
      relation_t relation = read_and_write[to].compare(&(op_from -> read_and_write)[from]);
      switch (relation)
      {
      case diff_mat: case same_mat_diff_branch: case same_node_no_overlap: case same_node_different_temp:
        break;
      case same_branch_diff_node: case same_node_overlapped: case same_index:
        dep |= (int) dependency_t::output_dep;
      }
    }
    else
    {
      relation_t relation = read_only[to - rw_to].compare(&(op_from -> read_and_write)[from]);
      switch (relation)
      {
      case diff_mat: case same_mat_diff_branch: case same_node_no_overlap: case same_node_different_temp:
        break;
      case same_branch_diff_node: case same_node_overlapped: case same_index:
        dep |= (int) dependency_t::flow_dep;
      }
    }
  }

  for (int i = 0; i < ro_from * rw_to; i++)
  {
    const int to = i / ro_from, from = i - to * ro_from;
    relation_t relation = read_and_write[to].compare(&(op_from -> read_only)[from]);
    switch (relation)
    {
    case diff_mat: case same_mat_diff_branch: case same_node_no_overlap: case same_node_different_temp:
      break;
    case same_branch_diff_node: case same_node_overlapped: case same_index:
      dep |= (int) dependency_t::anti_dep;
    }
  }

  return (dependency_t) dep;
}

dependency_t Operation::checkDependencyTo (const Operation& op_to) const
{ return op_to.checkDependencyFrom(*this); }
