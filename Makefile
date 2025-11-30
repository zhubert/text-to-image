# Platform detection
UNAME := $(shell uname -s)
ROCM_AVAILABLE := $(shell command -v rocminfo 2>/dev/null)
NVIDIA_AVAILABLE := $(shell command -v nvidia-smi 2>/dev/null)

# Allow override: make install BACKEND=rocm|cuda|cpu|mps
BACKEND ?= auto

.PHONY: install install-deps help

help:
	@echo "Usage: make install [BACKEND=auto|rocm|cuda|cpu|mps]"
	@echo ""
	@echo "Targets:"
	@echo "  install      Install PyTorch and project dependencies"
	@echo "  help         Show this help message"
	@echo ""
	@echo "Backends:"
	@echo "  auto         Auto-detect based on platform and available hardware (default)"
	@echo "  rocm         Force AMD ROCm (Linux only)"
	@echo "  cuda         Force NVIDIA CUDA (Linux only)"
	@echo "  cpu          Force CPU-only"
	@echo "  mps          Force Apple Metal (macOS only)"
	@echo ""
	@echo "Detected: UNAME=$(UNAME) ROCM=$(if $(ROCM_AVAILABLE),yes,no) NVIDIA=$(if $(NVIDIA_AVAILABLE),yes,no)"

install:
ifeq ($(BACKEND),rocm)
	@echo "Installing PyTorch for ROCm..."
	uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
else ifeq ($(BACKEND),cuda)
	@echo "Installing PyTorch for CUDA..."
	uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
else ifeq ($(BACKEND),cpu)
	@echo "Installing PyTorch for CPU..."
	uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
else ifeq ($(BACKEND),mps)
	@echo "Installing PyTorch for macOS (MPS)..."
	uv pip install torch torchvision
else
	# Auto-detect
  ifeq ($(UNAME),Darwin)
	@echo "Detected macOS, installing PyTorch with MPS support..."
	uv pip install torch torchvision
  else ifeq ($(UNAME),Linux)
    ifdef ROCM_AVAILABLE
	@echo "Detected Linux with ROCm, installing PyTorch for ROCm..."
	uv pip install torch torchvision --index-url https://download.pytorch.org/whl/rocm6.4
    else ifdef NVIDIA_AVAILABLE
	@echo "Detected Linux with NVIDIA GPU, installing PyTorch for CUDA..."
	uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
    else
	@echo "Detected Linux (no GPU), installing PyTorch for CPU..."
	uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    endif
  else
	@echo "Unknown platform, installing PyTorch for CPU..."
	uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  endif
endif
	@echo "Installing project dependencies..."
	uv pip install -e .
	@echo "Done!"
