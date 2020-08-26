
#pragma once
#ifndef _Dense
#define _Dense

#include <matrix/Element.cuh>

class Dense : public Element {
private:
  int m;
  int n;
  int ld;

  real_t* elements;

public:

  Dense(const int m, const int n, const int ld = 0);

  Dense(const int m, const int n, const int abs_y, const int abs_x, const int ld = 0);

  virtual ~Dense() override;

  virtual int getRowDimension() const override;

  virtual int getColumnDimension() const override;

  virtual int getLeadingDimension() const override;

  virtual real_t getElement(const int i, const int j) const override;

  real_t* getElements() const;

  real_t* getElements(const int offset) const;

  virtual void loadBinary(ifstream& stream) override;

  virtual void print(vector<int>& indices, vector<int>& config) const override;

  real_t sqrSum() const;

  real_t L2Error(const Dense& A) const;

  /*static h_ops_tree * generateOps_GETRF (const h_index * self, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_TRSML (const h_index * self, const Dense * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_TRSML (const h_index * self, const LowRank * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_TRSML (const h_index * self, const Hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_TRSML (const h_index * self, const Element * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_TRSMR (const h_index * self, const Dense * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_TRSMR (const h_index * self, const LowRank * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_TRSMR (const h_index * self, const Hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_TRSMR (const h_index * self, const Element * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_ACCM (const h_index * self, const h_index * index_tmp_lr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const Dense * A, const h_index * index_a, const Dense * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const LowRank * A, const h_index * index_a, const Dense * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const Hierarchical * A, const h_index * index_a, const Dense * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const Element * A, const h_index * index_a, const Dense * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const Dense * A, const h_index * index_a, const LowRank * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const LowRank * A, const h_index * index_a, const LowRank * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const Hierarchical * A, const h_index * index_a, const LowRank * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const Element * A, const h_index * index_a, const LowRank * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const Dense * A, const h_index * index_a, const Hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const LowRank * A, const h_index * index_a, const Hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const Hierarchical * A, const h_index * index_a, const Hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const Element * A, const h_index * index_a, const Hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const Dense * A, const h_index * index_a, const Element * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const LowRank * A, const h_index * index_a, const Element * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const Hierarchical * A, const h_index * index_a, const Element * B, const h_index * index_b, dev_temp * tmp_mngr);

  static h_ops_tree * generateOps_GEMM (const h_index * self, const Element * A, const h_index * index_a, const Element * B, const h_index * index_b, dev_temp * tmp_mngr);

  */


};


#endif