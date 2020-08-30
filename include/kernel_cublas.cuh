
#pragma once
#ifndef _KERNEL_CUH
#define _KERNEL_CUH

void kernel_cublas(const int n_streams, const int n_insts, const vector<int>* insts, vector<double*> ptrs);


#endif