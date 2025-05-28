# Setup script for Flash Attention implementations
import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing requirements...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA available: {torch.cuda.get_device_name()}")
            print(f"CUDA version: {torch.version.cuda}")
            return True
        else:
            print("CUDA not available")
            return False
    except ImportError:
        print("PyTorch not installed")
        return False

def check_triton():
    """Check Triton availability"""
    try:
        import triton
        print(f"Triton version: {triton.__version__}")
        return True
    except ImportError:
        print("Triton not available")
        return False

def run_tests():
    """Run test suite"""
    print("Running tests...")
    subprocess.check_call([sys.executable, "-m", "pytest", "test_attention.py", "-v"])

def run_benchmark():
    """Run benchmark suite"""
    print("Running benchmark...")
    subprocess.check_call([sys.executable, "benchmark.py"])

def main():
    print("Flash Attention Setup")
    print("=" * 30)
    
    # Install requirements
    install_requirements()
    
    # Check dependencies
    cuda_available = check_cuda()
    triton_available = check_triton()
    
    if not cuda_available:
        print("Warning: CUDA not available. Only CPU testing will be possible.")
    
    if not triton_available:
        print("Error: Triton not available. Please install triton.")
        return
    
    print("\nSetup complete!")
    
    # Ask user what to do
    while True:
        print("\nWhat would you like to do?")
        print("1. Run tests")
        print("2. Run benchmark")
        print("3. Exit")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            try:
                run_tests()
            except subprocess.CalledProcessError:
                print("Tests failed!")
        elif choice == "2":
            try:
                run_benchmark()
            except subprocess.CalledProcessError:
                print("Benchmark failed!")
        elif choice == "3":
            break
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()
