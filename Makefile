# Makefile for Flash Attention CUDA kernels

# CUDA compiler
NVCC = nvcc

# CUDA compiler flags
NVCC_FLAGS = -shared -Xcompiler -fPIC -O3 --use_fast_math
NVCC_FLAGS += -gencode arch=compute_70,code=sm_70
NVCC_FLAGS += -gencode arch=compute_75,code=sm_75
NVCC_FLAGS += -gencode arch=compute_80,code=sm_80
NVCC_FLAGS += -gencode arch=compute_86,code=sm_86

# Include directories
INCLUDES = -I$(CUDA_HOME)/include

# Library directories and libraries
LIBS = -L$(CUDA_HOME)/lib64 -lcuda -lcublas

# Source and target files
SRC = flash_attention.cu
TARGET = libflash_attention.so

# Default target
all: $(TARGET)

# Compile CUDA kernels into shared library
$(TARGET): $(SRC)
	$(NVCC) $(NVCC_FLAGS) $(INCLUDES) $(LIBS) -o $@ $<

# Clean build artifacts
clean:
	rm -f $(TARGET)

# Test compilation
test: $(TARGET)
	python test_flash_attention_cuda.py

# Install dependencies
install-deps:
	pip install torch torchvision pytest numpy

# Help target
help:
	@echo "Available targets:"
	@echo "  all          - Build the shared library (default)"
	@echo "  clean        - Remove build artifacts"
	@echo "  test         - Build and run tests"
	@echo "  install-deps - Install Python dependencies"
	@echo "  help         - Show this help message"

.PHONY: all clean test install-deps help
