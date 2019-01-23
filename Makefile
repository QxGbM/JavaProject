.SUFFIXES: .cpp .cu

CFLAGS		+= -std=c++11 -ggdb3 -O3 -fopenmp -I. -Wall -Wfatal-errors
NVCCFLAGS	+= -std=c++11 -I./include -arch sm_60 -Xcompiler "-ggdb3 -fopenmp -Wall -Wfatal-errors"
LDFLAGS 	+= -lm -ldl -lstdc++ -lpthread -lblas -llapacke -lcuda -lcudart 

CXX 		= g++
NVCC 		= nvcc

USE_CUB		= TRUE

#USE_KBLAS 	= TRUE
#USE_MKL	= TRUE

ifdef USE_CUB

CUB_ROOT	= /home/qxm/cub
NVCCFLAGS	+= -I$(CUB_ROOT)

endif

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

BIN = ./bin

.cpp.o:
	mkdir --parents $(BIN)
	$(CXX) $(CFLAGS) -c $? -o $(BIN)/$@

.cu.o:
	mkdir --parents $(BIN)
	$(NVCC) $(NVCCFLAGS) -c $? -o $(BIN)/$@

all:
	make gpu_lu

gpu_lu: $(BIN)/dense_lu_test.o
	$(CXX) $? $(LDFLAGS)
	./a.out

pivot: $(BIN)/pivot.o
	$(CXX) $? $(LDFLAGS)
	./a.out

svd: $(BIN)/svd.o
	$(CXX) $? $(LDFLAGS)
	./a.out

clean:
	$(RM) *.o *.a *.out *.xml
	$(RM) -r $(BIN)
