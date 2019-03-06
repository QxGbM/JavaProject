.SUFFIXES: .cu

#USE_MKL = TRUE

NVCC = /usr/local/cuda/bin/nvcc

BIN	= ./bin
INCLUDE	= ./include

ARCH += -gencode=arch=compute_50,code=sm_50
ARCH += -gencode=arch=compute_52,code=sm_52
ARCH += -gencode=arch=compute_60,code=sm_60
ARCH += -gencode=arch=compute_61,code=sm_61
ARCH += -gencode=arch=compute_61,code=compute_61
ARCH += -gencode=arch=compute_70,code=sm_70
ARCH += -gencode=arch=compute_70,code=compute_70

NVCCFLAGS += -std=c++11 -I$(INCLUDE) $(ARCH) -rdc=true -O2
LDFLAGS  += -lm -lstdc++ -lcuda -lcudart -L/usr/lib/x86_64-linux-gnu $(ARCH)

ifdef USE_MKL

LDFLAGS += -lmkl_intel_lp64 -lmkl_sequential -lmkl_core

endif

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
	$(NVCC) $(BIN)/$? $(LDFLAGS) -o $(BIN)/$@
	./$(BIN)/$@

test: test.o
	$(NVCC) $(BIN)/$? $(LDFLAGS) -o $(BIN)/$@
	./$(BIN)/$@

clean:
	$(RM) -r $(BIN)
