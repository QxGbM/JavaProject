## 1. Partition Large LR when gemm with H.

When doing gemm with H, different portions of LR can have different ranks, or even become dense in some portions.

LR needs to be partitioned before generating the h_ops_tree.

Update: done (2019 / 5 / 17)

## 2. Finish up dev-gemm including LR.

a. Compression (QR) after concatenation.

b. Handle temporary data from qr and svd properly.

## 3. Piplelined Multiple Kernel Launches.

Make dag flatten only some portions of the tree.

## More

TBA
