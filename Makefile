.SUFFIXES: .cpp .cu

CFLAGS		+= -std=c++11 -ggdb3 -O3 -fopenmp -I. -Wall -Wfatal-errors
NVCCFLAGS	+= -std=c++11 -I. -arch sm_60 -Xcompiler "-ggdb3 -fopenmp -Wall -Wfatal-errors"
LDFLAGS 	+= -lm -ldl -lstdc++ -lpthread -lblas -llapacke -lcuda -lcudart 

CXX 		= g++
NVCC 		= nvcc

#USE_KBLAS 	= TRUE
#USE_MKL	= TRUE

ifdef USE_KBLAS

KBLAS_ROOT	= /home/qxm/kblas-gpu

KBLAS_INCLUDE	= $(KBLAS_ROOT)/include
KBLAS_TESTING	= $(KBLAS_ROOT)/testing
NVCCFLAGS	+= -I$(KBLAS_INCLUDE) -I$(KBLAS_TESTING)

KBLAS_LIB	= $(KBLAS_ROOT)/lib
LDFLAGS		+= -L$(KBLAS_LIB) -lkblas-gpu -lcublas

endif

ifdef USE_MKL

LDFLAGS += -lmkl_intel_lp64 -lmkl_sequential -lmkl_core

endif

HELPERS = helper_functions.o cuda_helper_functions.o

.cpp.o:
	$(CXX) $(CFLAGS) -c $? -o $@

.cu.o:
	$(NVCC) $(NVCCFLAGS) -c $? -o $@

all:
	make gpu_lu

gpu_lu: dense_lu_test.o dense_lu.o gpu_lu.o $(HELPERS)
	$(CXX) $? $(LDFLAGS)
	./a.out

pivot: pivot.o $(HELPERS)
	$(CXX) $? $(LDFLAGS)
	./a.out

svd: svd.o $(HELPERS)
	$(CXX) $? $(LDFLAGS)
	./a.out

clean:
	(cd cuda_src && $(RM) -r *.o *.a *.out *.xml)
