.SUFFIXES: .cu

#USE_MKL = TRUE

NVCC = /usr/local/cuda-10.1/bin/nvcc -ccbin g++ -std=c++11 -m64

HOME_DIR = ./

PSPL_DIR = $(HOME_DIR)
BIN_DIR = $(HOME_DIR)bin/
INCLUDE_DIR = $(PSPL_DIR)include/

ARCH += -gencode arch=compute_30,code=sm_30
ARCH += -gencode arch=compute_35,code=sm_35
ARCH += -gencode arch=compute_37,code=sm_37
ARCH += -gencode arch=compute_50,code=sm_50
ARCH += -gencode arch=compute_52,code=sm_52
ARCH += -gencode arch=compute_60,code=sm_60
ARCH += -gencode arch=compute_61,code=sm_61
ARCH += -gencode arch=compute_70,code=sm_70
ARCH += -gencode arch=compute_75,code=sm_75
ARCH += -gencode arch=compute_75,code=compute_75

NVCCFLAGS = -rdc=true -O3 -I$(INCLUDE_DIR) $(ARCH) 
LDFLAGS  = -lstdc++ -lm -lcuda -lcudart -L/usr/lib/x86_64-linux-gnu $(ARCH)

ifdef USE_MKL

INCLUDE += 
LDFLAGS += -lmkl_intel_lp64 -lmkl_sequential -lmkl_core

endif

all:
	mkdir --parents $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -c $(PSPL_DIR)main.cu -o $(BIN_DIR)main.o
	$(NVCC) $(BIN_DIR)main.o $(LDFLAGS) -o $(BIN_DIR)main
	./$(BIN_DIR)main

clean:
	rm -r $(BIN_DIR)
