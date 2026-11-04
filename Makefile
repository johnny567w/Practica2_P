CUDA_ARCH ?= sm_75
NVCC      ?= nvcc
CXX       ?= g++
NVFLAGS    = -O3 -arch=$(CUDA_ARCH) -std=c++17 -Xcompiler -O2 -Isrc
COMMON     = src/conv_cpu.cpp src/conv_gpu.cu src/filters.cpp src/image_io.cpp src/validation.cpp

all: bin/benchmark bin/validate bin/filter bin/prepare_data

bin/benchmark: src/benchmark.cpp $(COMMON) | dirs
	$(NVCC) $(NVFLAGS) -o $@ $< $(COMMON)

bin/validate: src/validate.cpp $(COMMON) | dirs
	$(NVCC) $(NVFLAGS) -o $@ $< $(COMMON)

bin/filter: src/filter.cpp $(COMMON) | dirs
	$(NVCC) $(NVFLAGS) -o $@ $< $(COMMON)

bin/prepare_data: src/prepare_data.cpp src/image_io.cpp | dirs
	$(CXX) -O2 -std=c++17 -o $@ $^

dirs:
	mkdir -p bin data results/images

clean:
	rm -rf bin

.PHONY: all dirs clean
