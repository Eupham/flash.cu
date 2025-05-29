#!/bin/bash

# Flash Attention CUDA Kernel Setup and Test Script

set -e  # Exit on any error

echo "=========================================="
echo "Flash Attention CUDA Kernel Test Suite"
echo "=========================================="

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "Error: NVCC not found. Please ensure CUDA is installed and in PATH."
    exit 1
fi

# Check if Python is available
if ! command -v python &> /dev/null && ! command -v python3 &> /dev/null; then
    echo "Error: Python not found."
    exit 1
fi

# Use python3 if available, otherwise python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

echo "Using Python: $(which $PYTHON_CMD)"
echo "Using NVCC: $(which nvcc)"

# Check CUDA version
echo "CUDA Version: $(nvcc --version | grep release)"

# Install Python dependencies
echo
echo "Installing Python dependencies..."
$PYTHON_CMD -m pip install torch torchvision pytest numpy --quiet

# Check if PyTorch has CUDA support
echo
echo "Checking PyTorch CUDA support..."
$PYTHON_CMD -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')"

if ! $PYTHON_CMD -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    echo "Warning: PyTorch does not have CUDA support. Some tests may be skipped."
fi

# Compile CUDA kernels
echo
echo "Compiling CUDA kernels..."
if make clean && make; then
    echo "✓ CUDA kernels compiled successfully"
else
    echo "✗ Failed to compile CUDA kernels"
    echo "Make sure you have:"
    echo "  - CUDA toolkit installed"
    echo "  - Appropriate GPU architecture support"
    echo "  - CUDA_HOME environment variable set (if needed)"
    exit 1
fi

# Check if library was created
if [ ! -f "libflash_attention.so" ]; then
    echo "✗ Shared library not found after compilation"
    exit 1
fi

echo "✓ Shared library created: libflash_attention.so"

# Run basic functionality test
echo
echo "Running basic functionality tests..."
if $PYTHON_CMD test_flash_attention_cuda.py; then
    echo "✓ Basic tests passed"
else
    echo "✗ Basic tests failed"
    exit 1
fi

# Run full test suite with pytest if available
echo
echo "Running full test suite..."
if command -v pytest &> /dev/null; then
    if pytest test_flash_attention_cuda.py -v; then
        echo "✓ Full test suite passed"
    else
        echo "✗ Some tests failed"
        exit 1
    fi
else
    echo "pytest not available, skipping extended tests"
    echo "Install pytest with: pip install pytest"
fi

echo
echo "=========================================="
echo "✓ All tests completed successfully!"
echo "=========================================="
echo
echo "You can now use the Flash Attention kernels:"
echo "  python test_flash_attention_cuda.py  # Run basic tests"
echo "  pytest test_flash_attention_cuda.py -v  # Run full test suite"
echo
echo "The compiled library is available at: libflash_attention.so"
