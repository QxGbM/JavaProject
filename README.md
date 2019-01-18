# Pastel-Palettes

**P**erformance **A**imed **S**of**t**war**e L**ibrary * G**P**U **A**cce**le**ra**t**ed Rou**t**in**es**

Abbr: パスパレ (pa-su-pa-re)

Named after: Virtual Band "Pastel * Palettes" from Bang Dream! Girls Band Party.

## Maintained by

QxGbM, qxm28@case.edu

## 说明

这是我现在的项目（的自测代码）。真正的项目是用cuda和nvcc（nvidia的c/c++编译器）实装完整矩阵，低阶矩阵，完整和低阶混合的阶层矩阵的lu分解。

完整矩阵：【【1 2 3】 【4 5 6】 【7 8 9】】

低阶矩阵：完整矩阵的奇异值分解（svd），并通过砍掉过小的奇异值输入（sigma矩阵中的最后数个对角线输入）来降低数据存储量，这是对完整矩阵的一种估算算法

阶层矩阵：可以是【【阶层 阶层】【阶层 阶层】】或者把其中任意一个阶层矩阵换成完整矩阵或者低阶矩阵，如此递归定义可以让一个矩阵同时保有数个子完整矩阵和低阶矩阵，保证了估算准确度的同时也降低了数据存储量

lu分解：一个矩阵变成l * u的形式，l是下三角矩阵（对角线右上输入全0，对角线输入全1），u是上三角（对角线左下输入全0）。这个分解可以用来解Ax=y
