# Pastel-Palettes

**P**erformance **A**imed **S**of**t**war**e L**ibrary * G**P**U **A**cce**le**ra**t**ed Rou**t**in**es**

Abbr: パスパレ (pa-su-pa-re)

Named after: Virtual Band "Pastel * Palettes" from Bang Dream! Girls Band Party.

## Maintained by

QxGbM, qxm28@case.edu

## Terminology Explanations

Dense: All matrix elements are stored.

Low Rank: A matrix stored as a product of 2 or 3 smaller sized matrices.

Hierarchical: A tree structure that has a Dense, a Low Rank, or another tree as its leaf. Does not store individual elements.

LU Decomposition: A matrix decomposition that generates a lower and upper matrices, where A = LU.

H-op: A matrix operation routine that happens between different nodes in the hierarchy.

- GETRF: LU decomposition on 1 node.

- TRSM: Triangular Solve on decompose L/U against another node.

- GEMM: General Matrix Multiplication on node A, B, and C, so that A = A - B * C.

- ACCM: Accumulation on node A, B, so that A = A + B.

DAG: Direct Acyclic Graph. Connections exist when there are data dependencies between 2 h-ops.

Instruction for GPU: A list of integers to retrieve data pointers and matrix dimensions.

## H - LU Behavior

1. Read H-structure and generate a tree of h-ops.

2. Create DAG from the tree of h-ops.

3. Schedule the DAG according to the number of workers.

4. Interpret the DAG and Schedule into lists of Instructions.

5. GPU Kernel executes the Instructions.

## Others

More information on this project in the "docs" folder.
