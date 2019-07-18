.SUFFIXES: .cu

NVCC = nvcc 
NVCC += -ccbin g++ -std=c++14

HOME_DIR = ./

PSPL_DIR = $(HOME_DIR)
BIN_DIR = $(HOME_DIR)bin/
INCLUDE_DIR = $(PSPL_DIR)include/

ARCH += -gencode arch=compute_60,code=sm_60
ARCH += -gencode arch=compute_61,code=sm_61
ARCH += -gencode arch=compute_70,code=sm_70
ARCH += -gencode arch=compute_75,code=sm_75
ARCH += -gencode arch=compute_75,code=compute_75

NVCCFLAGS = -maxrregcount=128 --machine 64 -rdc=true -O3 
NVCCFLAGS += -I$(INCLUDE_DIR) 
NVCCFLAGS += $(ARCH) 
NVCCFLAGS += -Xcompiler "-fopenmp"

LDFLAGS = -lstdc++ -lm -lcuda -lcudart -Xcompiler "-fopenmp"
LDFLAGS += -L/usr/lib/x86_64-linux-gnu $(ARCH)

#USE_MKL = TRUE

ifdef USE_MKL

INCLUDE += 
LDFLAGS += -lmkl_intel_lp64 -lmkl_sequential -lmkl_core

endif

all:
	make compile
	make data
	make tests

compile:
	mkdir --parents $(BIN_DIR)
	$(NVCC) $(NVCCFLAGS) -c $(PSPL_DIR)main.cu -o $(BIN_DIR)main.o
	$(NVCC) $(BIN_DIR)main.o $(LDFLAGS) -o $(BIN_DIR)main

data:
	java -Xmx16g -jar Pastel-Palettes-1.0-SNAPSHOT.jar -level=1 -dim=1024 -h=test_1k -d=ref_1k
	java -Xmx16g -jar Pastel-Palettes-1.0-SNAPSHOT.jar -level=2 -dim=2048 -h=test_2k -d=ref_2k
	java -Xmx16g -jar Pastel-Palettes-1.0-SNAPSHOT.jar -level=3 -dim=4096 -h=test_4k -d=ref_4k
	java -Xmx16g -jar Pastel-Palettes-1.0-SNAPSHOT.jar -level=4 -dim=8192 -h=test_8k -skipd
	java -Xmx16g -jar Pastel-Palettes-1.0-SNAPSHOT.jar -level=5 -dim=16384 -h=test_16k -skipd
	java -Xmx16g -jar Pastel-Palettes-1.0-SNAPSHOT.jar -level=6 -dim=32768 -h=test_32k -skipd

tests:
	$(BIN_DIR)main -test=test_1k -ref=ref_1k
	$(BIN_DIR)main -test=test_2k -ref=ref_2k
	$(BIN_DIR)main -test=test_4k -ref=ref_4k
	$(BIN_DIR)main -test=test_8k -noref
	$(BIN_DIR)main -test=test_16k -noref
	$(BIN_DIR)main -test=test_32k -noref

clean:
	rm -rf $(BIN_DIR)
