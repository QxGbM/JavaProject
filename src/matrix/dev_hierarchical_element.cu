
#include <definitions.cuh>
#include <matrix/dev_dense.cuh>
#include <matrix/dev_low_rank.cuh>
#include <matrix/dev_hierarchical.cuh>
#include <matrix/dev_hierarchical_element.cuh>
#include <h_ops/dev_hierarchical_index.cuh>
#include <h_ops/dev_hierarchical_ops.cuh>
#include <h_ops/dev_hierarchical_ops_tree.cuh>
#include <dev_temp.cuh>


dev_h_element::dev_h_element (void * element_in, const element_t type_in, const int nx, const int ny)
{
  element = element_in;
  type = type_in;
  block_x = nx > 0 ? nx : getNx();
  block_y = ny > 0 ? ny : getNy();
  abs_x = 0;
  abs_y = 0;
}

dev_h_element::~dev_h_element ()
{ 
  dev_dense *d = getElementDense();
  dev_low_rank *lr = getElementLowRank();
  dev_hierarchical *h = getElementHierarchical();

  if (d != nullptr) { delete d; }
  if (lr != nullptr) { delete lr; }
  if (h != nullptr) { delete h; }
}

dev_dense * dev_h_element::getElementDense() const
{
  return (type == dense) ? ((dev_dense *) element) : nullptr;
}

dev_low_rank * dev_h_element::getElementLowRank() const
{
  return (type == low_rank) ? ((dev_low_rank *) element) : nullptr;
}

dev_hierarchical * dev_h_element::getElementHierarchical() const
{
  return (type == hierarchical) ? ((dev_hierarchical *) element) : nullptr;
}

element_t dev_h_element::getType() const
{ 
  return type; 
}

int dev_h_element::getNx() const
{
  const dev_dense *d = getElementDense();
  const dev_low_rank *lr = getElementLowRank();
  const dev_hierarchical *h = getElementHierarchical();

  if (d != nullptr)
  { return d -> getNx(); }
  if (lr != nullptr)
  { return lr -> getNx(); }
  if (h != nullptr)
  { return h -> getNx_abs(); }

  return 0;
}

int dev_h_element::getNy() const
{
  const dev_dense *d = getElementDense();
  const dev_low_rank *lr = getElementLowRank();
  const dev_hierarchical *h = getElementHierarchical();

  if (d != nullptr)
  { return d -> getNy(); }
  if (lr != nullptr)
  { return lr -> getNy(); }
  if (h != nullptr)
  { return h -> getNy_abs(); }

  return 0;
}

int dev_h_element::getLd() const
{
  const dev_dense *d = getElementDense();

  if (d != nullptr)
  { return d -> getLd(); }

  return 0;
}

int dev_h_element::getRank() const
{
  const dev_low_rank *lr = getElementLowRank();

  if (lr != nullptr)
  { return lr -> getRank(); }

  return 0;
}

void dev_h_element::setElement (void * element_in, element_t type_in)
{
  element = element_in;
  type = type_in;
}

real_t dev_h_element::getElement (const int y_in, const int x_in) const
{
  const dev_dense *d = getElementDense();
  const dev_low_rank *lr = getElementLowRank();
  const dev_hierarchical *h = getElementHierarchical();

  if (d != nullptr)
  { return (d -> getElements())[y_in * (d -> getLd()) + x_in]; }
  if (lr != nullptr)
  { return lr -> getElement(y_in, x_in); }
  if (h != nullptr)
  { return h -> getElement_abs(y_in, x_in); }

  return 0;
}

void dev_h_element::setAbs (const int abs_y_in, const int abs_x_in)
{ abs_y = abs_y_in; abs_x = abs_x_in; }

void dev_h_element::getAbs (int * abs_y_out, int * abs_x_out)
{ * abs_y_out = abs_y; * abs_x_out = abs_x; }

dev_dense * dev_h_element::convertToDense() const
{
  const dev_dense *d = getElementDense();
  const dev_low_rank *lr = getElementLowRank();
  const dev_hierarchical *h = getElementHierarchical();

  if (d != nullptr)
  { return (dev_dense *) d; }
  if (lr != nullptr)
  { return lr -> convertToDense(); }
  if (h != nullptr)
  { return h -> convertToDense(); }

  return nullptr;
}

h_ops_tree * dev_h_element::generateOps_GETRF (const h_index * self, dev_temp * tmp_mngr) const
{
  const dev_dense*d = getElementDense();
  const dev_low_rank*lr = getElementLowRank();
  const dev_hierarchical *h = getElementHierarchical();

  if (d != nullptr)
  { return d -> generateOps_GETRF(self, tmp_mngr); }
  if (lr != nullptr)
  { return lr -> generateOps_GETRF(self, tmp_mngr); }
  if (h != nullptr)
  { return h -> generateOps_GETRF(self, tmp_mngr); }

  return nullptr;
}

h_ops_tree * dev_h_element::generateOps_TRSML (const h_index * self, const dev_dense * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  const dev_dense*d = getElementDense();
  const dev_low_rank*lr = getElementLowRank();
  const dev_hierarchical *h = getElementHierarchical();

  if (d != nullptr)
  { return d -> generateOps_TRSML(self, B, index_b, tmp_mngr); }
  if (lr != nullptr)
  { return lr -> generateOps_TRSML(self, B, index_b, tmp_mngr); }
  if (h != nullptr)
  { return h -> generateOps_TRSML(self, B, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * dev_h_element::generateOps_TRSML (const h_index * self, const dev_low_rank * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  const dev_dense*d = getElementDense();
  const dev_low_rank*lr = getElementLowRank();
  const dev_hierarchical *h = getElementHierarchical();

  if (d != nullptr)
  { return d -> generateOps_TRSML(self, B, index_b, tmp_mngr); }
  if (lr != nullptr)
  { return lr -> generateOps_TRSML(self, B, index_b, tmp_mngr); }
  if (h != nullptr)
  { return h -> generateOps_TRSML(self, B, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * dev_h_element::generateOps_TRSML (const h_index * self, const dev_hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  const dev_dense*d = getElementDense();
  const dev_low_rank*lr = getElementLowRank();
  const dev_hierarchical *h = getElementHierarchical();

  if (d != nullptr)
  { return d -> generateOps_TRSML(self, B, index_b, tmp_mngr); }
  if (lr != nullptr)
  { return lr -> generateOps_TRSML(self, B, index_b, tmp_mngr); }
  if (h != nullptr)
  { return h -> generateOps_TRSML(self, B, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * dev_h_element::generateOps_TRSML (const h_index * self, const dev_h_element * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  const dev_dense*d = getElementDense();
  const dev_low_rank*lr = getElementLowRank();
  const dev_hierarchical *h = getElementHierarchical();

  if (d != nullptr)
  { return d -> generateOps_TRSML(self, B, index_b, tmp_mngr); }
  if (lr != nullptr)
  { return lr -> generateOps_TRSML(self, B, index_b, tmp_mngr); }
  if (h != nullptr)
  { return h -> generateOps_TRSML(self, B, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * dev_h_element::generateOps_TRSMR (const h_index * self, const dev_dense * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  const dev_dense*d = getElementDense();
  const dev_low_rank*lr = getElementLowRank();
  const dev_hierarchical *h = getElementHierarchical();

  if (d != nullptr)
  { return d -> generateOps_TRSMR(self, B, index_b, tmp_mngr); }
  if (lr != nullptr)
  { return lr -> generateOps_TRSMR(self, B, index_b, tmp_mngr); }
  if (h != nullptr)
  { return h -> generateOps_TRSMR(self, B, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * dev_h_element::generateOps_TRSMR (const h_index * self, const dev_low_rank * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  const dev_dense*d = getElementDense();
  const dev_low_rank*lr = getElementLowRank();
  const dev_hierarchical *h = getElementHierarchical();

  if (d != nullptr)
  { return d -> generateOps_TRSMR(self, B, index_b, tmp_mngr); }
  if (lr != nullptr)
  { return lr -> generateOps_TRSMR(self, B, index_b, tmp_mngr); }
  if (h != nullptr)
  { return h -> generateOps_TRSMR(self, B, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * dev_h_element::generateOps_TRSMR (const h_index * self, const dev_hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  const dev_dense*d = getElementDense();
  const dev_low_rank*lr = getElementLowRank();
  const dev_hierarchical *h = getElementHierarchical();

  if (d != nullptr)
  { return d -> generateOps_TRSMR(self, B, index_b, tmp_mngr); }
  if (lr != nullptr)
  { return lr -> generateOps_TRSMR(self, B, index_b, tmp_mngr); }
  if (h != nullptr)
  { return h -> generateOps_TRSMR(self, B, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * dev_h_element::generateOps_TRSMR (const h_index * self, const dev_h_element * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  const dev_dense*d = getElementDense();
  const dev_low_rank*lr = getElementLowRank();
  const dev_hierarchical *h = getElementHierarchical();

  if (d != nullptr)
  { return d -> generateOps_TRSMR(self, B, index_b, tmp_mngr); }
  if (lr != nullptr)
  { return lr -> generateOps_TRSMR(self, B, index_b, tmp_mngr); }
  if (h != nullptr)
  { return h -> generateOps_TRSMR(self, B, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * dev_h_element::generateOps_ACCM (const h_index * self, const h_index * index_tmp_lr) const
{
  const dev_dense*d = getElementDense();
  const dev_low_rank*lr = getElementLowRank();
  const dev_hierarchical *h = getElementHierarchical();

  if (d != nullptr)
  { return d -> generateOps_ACCM(self, index_tmp_lr); }
  if (lr != nullptr)
  { return lr -> generateOps_ACCM(self, index_tmp_lr); }
  if (h != nullptr)
  { return h -> generateOps_ACCM(self, index_tmp_lr); }

  return nullptr;
}


h_ops_tree * dev_h_element::generateOps_GEMM (const h_index * self, const dev_dense * A, const h_index * index_a, const dev_dense * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  const dev_dense*d = getElementDense();
  const dev_low_rank*lr = getElementLowRank();
  const dev_hierarchical *h = getElementHierarchical();

  if (d != nullptr)
  { return d -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
  if (lr != nullptr)
  { return lr -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
  if (h != nullptr)
  { return h -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * dev_h_element::generateOps_GEMM (const h_index * self, const dev_dense * A, const h_index * index_a, const dev_low_rank * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  const dev_dense*d = getElementDense();
  const dev_low_rank*lr = getElementLowRank();
  const dev_hierarchical *h = getElementHierarchical();

  if (d != nullptr)
  { return d -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
  if (lr != nullptr)
  { return lr -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
  if (h != nullptr)
  { return h -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }

  return nullptr;
}
  
h_ops_tree * dev_h_element::generateOps_GEMM (const h_index * self, const dev_dense * A, const h_index * index_a, const dev_hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  const dev_dense*d = getElementDense();
  const dev_low_rank*lr = getElementLowRank();
  const dev_hierarchical *h = getElementHierarchical();

  if (d != nullptr)
  { return d -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
  if (lr != nullptr)
  { return lr -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
  if (h != nullptr)
  { return h -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * dev_h_element::generateOps_GEMM (const h_index * self, const dev_dense * A, const h_index * index_a, const dev_h_element * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  const dev_dense*d = getElementDense();
  const dev_low_rank*lr = getElementLowRank();
  const dev_hierarchical *h = getElementHierarchical();

  if (d != nullptr)
  { return d -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
  if (lr != nullptr)
  { return lr -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
  if (h != nullptr)
  { return h -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * dev_h_element::generateOps_GEMM (const h_index * self, const dev_low_rank * A, const h_index * index_a, const dev_dense * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  const dev_dense*d = getElementDense();
  const dev_low_rank*lr = getElementLowRank();
  const dev_hierarchical *h = getElementHierarchical();

  if (d != nullptr)
  { return d -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
  if (lr != nullptr)
  { return lr -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
  if (h != nullptr)
  { return h -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * dev_h_element::generateOps_GEMM (const h_index * self, const dev_low_rank * A, const h_index * index_a, const dev_low_rank * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  const dev_dense*d = getElementDense();
  const dev_low_rank*lr = getElementLowRank();
  const dev_hierarchical *h = getElementHierarchical();

  if (d != nullptr)
  { return d -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
  if (lr != nullptr)
  { return lr -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
  if (h != nullptr)
  { return h -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * dev_h_element::generateOps_GEMM (const h_index * self, const dev_low_rank * A, const h_index * index_a, const dev_hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  const dev_dense*d = getElementDense();
  const dev_low_rank*lr = getElementLowRank();
  const dev_hierarchical *h = getElementHierarchical();

  if (d != nullptr)
  { return d -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
  if (lr != nullptr)
  { return lr -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
  if (h != nullptr)
  { return h -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * dev_h_element::generateOps_GEMM (const h_index * self, const dev_low_rank * A, const h_index * index_a, const dev_h_element * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  const dev_dense*d = getElementDense();
  const dev_low_rank*lr = getElementLowRank();
  const dev_hierarchical *h = getElementHierarchical();

  if (d != nullptr)
  { return d -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
  if (lr != nullptr)
  { return lr -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
  if (h != nullptr)
  { return h -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * dev_h_element::generateOps_GEMM (const h_index * self, const dev_hierarchical * A, const h_index * index_a, const dev_dense * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  const dev_dense*d = getElementDense();
  const dev_low_rank*lr = getElementLowRank();
  const dev_hierarchical *h = getElementHierarchical();

  if (d != nullptr)
  { return d -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
  if (lr != nullptr)
  { return lr -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
  if (h != nullptr)
  { return h -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * dev_h_element::generateOps_GEMM (const h_index * self, const dev_hierarchical * A, const h_index * index_a, const dev_low_rank * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  const dev_dense*d = getElementDense();
  const dev_low_rank*lr = getElementLowRank();
  const dev_hierarchical *h = getElementHierarchical();

  if (d != nullptr)
  { return d -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
  if (lr != nullptr)
  { return lr -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
  if (h != nullptr)
  { return h -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * dev_h_element::generateOps_GEMM (const h_index * self, const dev_hierarchical * A, const h_index * index_a, const dev_hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  const dev_dense*d = getElementDense();
  const dev_low_rank*lr = getElementLowRank();
  const dev_hierarchical *h = getElementHierarchical();

  if (d != nullptr)
  { return d -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
  if (lr != nullptr)
  { return lr -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
  if (h != nullptr)
  { return h -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * dev_h_element::generateOps_GEMM (const h_index * self, const dev_hierarchical * A, const h_index * index_a, const dev_h_element * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  const dev_dense*d = getElementDense();
  const dev_low_rank*lr = getElementLowRank();
  const dev_hierarchical *h = getElementHierarchical();

  if (d != nullptr)
  { return d -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
  if (lr != nullptr)
  { return lr -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
  if (h != nullptr)
  { return h -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * dev_h_element::generateOps_GEMM (const h_index * self, const dev_h_element * A, const h_index * index_a, const dev_dense * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  const dev_dense*d = getElementDense();
  const dev_low_rank*lr = getElementLowRank();
  const dev_hierarchical *h = getElementHierarchical();

  if (d != nullptr)
  { return d -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
  if (lr != nullptr)
  { return lr -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
  if (h != nullptr)
  { return h -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * dev_h_element::generateOps_GEMM (const h_index * self, const dev_h_element * A, const h_index * index_a, const dev_low_rank * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  const dev_dense*d = getElementDense();
  const dev_low_rank*lr = getElementLowRank();
  const dev_hierarchical *h = getElementHierarchical();

  if (d != nullptr)
  { return d -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
  if (lr != nullptr)
  { return lr -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
  if (h != nullptr)
  { return h -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * dev_h_element::generateOps_GEMM (const h_index * self, const dev_h_element * A, const h_index * index_a, const dev_hierarchical * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  const dev_dense*d = getElementDense();
  const dev_low_rank*lr = getElementLowRank();
  const dev_hierarchical *h = getElementHierarchical();

  if (d != nullptr)
  { return d -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
  if (lr != nullptr)
  { return lr -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
  if (h != nullptr)
  { return h -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }

  return nullptr;
}

h_ops_tree * dev_h_element::generateOps_GEMM (const h_index * self, const dev_h_element * A, const h_index * index_a, const dev_h_element * B, const h_index * index_b, dev_temp * tmp_mngr) const
{
  const dev_dense*d = getElementDense();
  const dev_low_rank*lr = getElementLowRank();
  const dev_hierarchical *h = getElementHierarchical();

  if (d != nullptr)
  { return d -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
  if (lr != nullptr)
  { return lr -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }
  if (h != nullptr)
  { return h -> generateOps_GEMM(self, A, index_a, B, index_b, tmp_mngr); }

  return nullptr;
}

cudaError_t dev_h_element::loadBinary (FILE * stream, const bool reverse_bytes)
{
  dev_dense*d = getElementDense();
  dev_low_rank*lr = getElementLowRank();
  dev_hierarchical *h = getElementHierarchical();

  if (d != nullptr)
  { return d -> loadBinary(stream, reverse_bytes); }
  if (lr != nullptr)
  { return lr -> loadBinary(stream, reverse_bytes); }
  if (h != nullptr)
  { return h -> loadBinary(stream, reverse_bytes); }

  return cudaErrorMissingConfiguration;
}

void * dev_h_element::readStructureFromFile (FILE * stream, element_t * type, const int shadow_rank)
{
  char * buf = new char[32];
  if (stream != nullptr && fgets(buf, 32, stream) > 0)
  {
    int ny, nx, rank;

    if (buf[0] == 'H')
    { 
      * type = hierarchical;
      sscanf(buf, "H %d %d\n", &ny, &nx);
      delete[] buf;

      dev_hierarchical* h = new dev_hierarchical(nx, ny);

      for (int i = 0; i < ny; i++) for (int j = 0; j < nx; j++)
      {
        element_t type;
        void * element = readStructureFromFile (stream, &type, shadow_rank);
        h -> setElement(element, type, j, i);
      }

      h -> updateOffsets();
      return h;
    }
    else if (buf[0] == 'D')
    { 
      * type = dense; 
      sscanf(buf, "D %d %d\n", &ny, &nx);
      delete[] buf;
      dev_dense * d = new dev_dense (nx, ny);
      d -> resizeShadow (shadow_rank);
      return d;
    }
    else if (buf[0] == 'L' && buf[1] == 'R')
    { 
      * type = low_rank; 
      sscanf(buf, "LR %d %d %d\n", &ny, &nx, &rank);
      delete[] buf;
      return new dev_low_rank (nx, ny, rank);
    }
    else
    { 
      * type = empty; 
      ny = nx = rank = 0;
      delete[] buf;
      return nullptr;
    }
  }
  else
  {
    printf("Error Reading from File.\n");
    delete[] buf;
    return nullptr;
  }

}


void dev_h_element::print (std :: vector <int> &indices) const
{
  const dev_dense *d = getElementDense();
  const dev_low_rank *lr = getElementLowRank();
  const dev_hierarchical *h = getElementHierarchical();

  std :: cout << "(" << abs_y << ", " << abs_x << ") ";
  for (int i : indices) 
  { std::cout << i << ' '; }

  if (d != nullptr) { d -> print(); }
  if (lr != nullptr) { lr -> print(); }
  if (h != nullptr) { h -> print(indices); } 
}


