#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GPU Availability Check Script
"""

import torch

print("=" * 80)
print("PyTorch GPU Configuration Check")
print("=" * 80)

print(f"\nPyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"cuDNN Version: {torch.backends.cudnn.version()}")
    print(f"\nNumber of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\n--- GPU {i} ---")
        print(f"Name: {torch.cuda.get_device_name(i)}")
        print(f"Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"Compute Capability: {torch.cuda.get_device_capability(i)}")
    
    # Set default device
    device = torch.device('cuda:0')
    print(f"\n✓ Using device: {device}")
    
    # Test tensor on GPU
    x = torch.rand(5, 5).to(device)
    print(f"\n✓ Successfully created tensor on GPU")
    print(f"Tensor device: {x.device}")
else:
    print("\n✗ CUDA is not available. Will use CPU instead.")
    device = torch.device('cpu')
    print(f"Using device: {device}")

print("\n" + "=" * 80)
