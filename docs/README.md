# Pastel-Palettes

**P**erformance **A**imed **S**of**t**war**e L**ibrary * G**P**U **A**cce**le**ra**t**ed Rou**t**in**es**

Abbr: パスパレ (pa-su-pa-re)

Named after: Virtual Band "Pastel * Palettes" from Bang Dream! Girls Band Party.

## Maintained by

QxGbM, qxm28@case.edu

## 说明

对hicma的补充。项目目标是用使用一个cuda内核函数解决h-matrix的LU分解问题。

完整矩阵(Dense)：
【【1 2 3】
  【4 5 6】 
  【7 8 9】】

低阶矩阵(Low_Rank)：完整矩阵的奇异值分解（svd分解），并通过砍掉过小的奇异值输入（sigma矩阵中经过排序后，最后数个最小的对角线输入）来降低数据存储量和矩阵链乘速度。
【 U . S . VT 】

阶层矩阵(Hierarchical)：树型结构矩阵，叶节点为完整或低阶矩阵。中间结点不保存数据。

lu分解：一个矩阵变成l * u的形式，l是下三角矩阵（对角线右上输入全0，对角线输入全1），u是上三角（对角线左下输入全0）。这个分解可以用来解Ax=y

# Behavior

GETRF D -> D is LU decomposed.

GETRF LR -> not allowed.

GETRF H -> { GETRF H|i,i; TRSML H|i,k H|i,i; TRSMR H|j,i H|i,i; GEMM H|j,k H|j,i H|i,k; GETRF H|i+1,i+1 ... }

TRSM D L/U(D) -> Overwrites D with L^-1 x D.

TRSM _ L/U(LR) -> not allowed.

TRSM H L/U(H) -> { TRSML H|i L|i,i; GEMM H|i+1 L|j,i H|i, TRSML H|i+1 L|i+1,i+1; ... }

TRSM D L/U(H) -> Partition D to match with H

TRSM LR L/U(_) -> L: TRSML UxS L; R: TRSMR V U;

GEMM D M1 M2 -> D = D - M1 x M2

GEMM LR M1 M2 -> US = US - M1 x M2 x VT

GEMM H M1 M2 -> { GEMM H|i,j M1|i M2|j }