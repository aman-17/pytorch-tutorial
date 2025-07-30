"""
Question 3: Memory-Efficient Gradient Checkpointing (Hard)

Implement gradient checkpointing for a deep residual network to reduce memory usage during
backpropagation. Gradient checkpointing trades computation for memory by not storing
intermediate activations and recomputing them during backward pass.

Your task: Implement a ResidualBlock with optional gradient checkpointing and a ResNet
that can toggle checkpointing on/off.

Requirements:
1. ResidualBlock with skip connections
2. Gradient checkpointing functionality using torch.utils.checkpoint
3. Compare memory usage with/without checkpointing
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_checkpoint=False):
        super(ResidualBlock, self).__init__()
        # TODO: Implement residual block with:
        # - Two conv layers with batch norm and relu
        # - Skip connection (with 1x1 conv if dimensions don't match)
        # - Store use_checkpoint flag
        pass
    
    def forward(self, x):
        # TODO: Implement forward pass
        # If use_checkpoint is True, use torch.utils.checkpoint.checkpoint
        # for the main branch computation
        pass

class CheckpointResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=10, use_checkpoint=False):
        super(CheckpointResNet, self).__init__()
        # TODO: Create a ResNet with:
        # - Initial conv layer
        # - Multiple residual blocks
        # - Global average pooling
        # - Final classifier
        pass
    
    def forward(self, x):
        # TODO: Implement forward pass through all layers
        pass

def compare_memory_usage():
    """Compare memory usage with and without gradient checkpointing"""
    # TODO: Create two identical networks (with/without checkpointing)
    # TODO: Run forward and backward pass on both
    # TODO: Measure and compare memory usage using torch.cuda.memory_allocated()
    pass

# Test your implementation
if __name__ == "__main__":
    # Test basic functionality
    block = ResidualBlock(64, 64, use_checkpoint=True)
    x = torch.randn(2, 64, 32, 32)
    output = block(x)
    print(f"Block output shape: {output.shape}")
    
    # Compare memory usage
    if torch.cuda.is_available():
        compare_memory_usage()
    else:
        print("CUDA not available - memory comparison skipped")