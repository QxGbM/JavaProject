# Current Contributing List

## 1. Partition Large LR when gemm with H

When doing gemm with H, different portions of LR can have different ranks, or even become dense in some portions.

LR needs to be partitioned before generating the h_ops_tree.

Update: done (2019 / 5 / 17)

## 2. Finish up dev-gemm including LR

a. Compression (QR) after concatenation.

b. Handle temporary data from qr and svd properly.

Update: done (2019 / 6 / 13)

## 3. Piplelined Multiple Kernel Launches

Make dag flatten only some portions of the tree.

Update: Necessary function is built, but not implemented yet (2019 / 6 / 25)

## 4. Java Compressor Rework

As a alternative to GPU compression, possible visualization in the future.

Update: done (2019 / 6 / 25)

## 5. Fixing device Functions

Device side GEMM has really unstable scaling. Consider changing to tiled versions.

GETRF and TRSMs assumed dimensions smaller than some shared memory size, can fail on some large inputs.

Update: done (2019 / 7 / 12)

## 6. Using flops to schedule DAG

Flops for each GPU instruction can differ significantly if the hierarchy is deep.

The current scheduling heuristic works well if each instruction has similar flops, but not good with hierarchy.

Update: done (2019 / 7 / 12)

## 7. Seperate files for faster compilation

As Described.

## More

TBA
