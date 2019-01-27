.SUFFIXES: .cpp .cu

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

CXX 		= g++
NVCC 		= nvcc

BIN	= ./bin
INCLUDE	= ./include

CFLAGS		+= -std=c++11 -ggdb3 -O3 -fopenmp -I$(INCLUDE) -Wall -Wfatal-errors
NVCCFLAGS	+= -std=c++11 -I$(INCLUDE) -arch=sm_60 -rdc=true -Xcompiler "-ggdb3 -fopenmp -Wall -Wfatal-errors"
LDFLAGS 	+= -lm -ldl -lstdc++ -lpthread -lblas -llapacke -lcuda -lcudart -arch=sm_60 -L/usr/lib/x86_64-linux-gnu

.cpp.o:
	mkdir --parents $(BIN)
	$(CXX) $(CFLAGS) -c $? -o $(BIN)/$@

.cu.o:
	mkdir --parents $(BIN)
	$(NVCC) $(NVCCFLAGS) -c $? -o $(BIN)/$@

all:
	make gpu_lu

gpu_lu: dense_lu_test.o
	$(NVCC) $(BIN)/$? $(LDFLAGS) -o $(BIN)/$@
	./$(BIN)/$@

pivot: pivot.o
	$(NVCC) $(BIN)/$? $(LDFLAGS) -o $(BIN)/$@
	./$(BIN)/$@

svd: svd.o
	$(CXX) $(BIN)/$? $(LDFLAGS) -o $(BIN)/$@
	./$(BIN)/$@

test: test.o
	$(NVCC) $(BIN)/$? $(LDFLAGS) -o $(BIN)/$@
	./$(BIN)/$@

clean:
	$(RM) *.o *.a *.out *.xml
	$(RM) -r $(BIN)
