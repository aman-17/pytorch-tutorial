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
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_checkpoint=False):
        super(ResidualBlock, self).__init__()
        
        # Two conv layers with batch norm and relu
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection (with 1x1 conv if dimensions don't match)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # Store use_checkpoint flag
        self.use_checkpoint = use_checkpoint
    
    def _forward_impl(self, x):
        """Implementation of the forward pass for checkpointing"""
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return out
    
    def forward(self, x):
        # If use_checkpoint is True, use torch.utils.checkpoint.checkpoint
        # for the main branch computation
        if self.use_checkpoint and self.training:
            # Use gradient checkpointing for the main branch
            out = checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            # Regular forward pass
            out = self._forward_impl(x)
        
        # Add skip connection
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class CheckpointResNet(nn.Module):
    def __init__(self, num_blocks, num_classes=10, use_checkpoint=False):
        super(CheckpointResNet, self).__init__()
        
        # Initial conv layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Multiple residual blocks
        self.layer1 = self._make_layer(64, 64, num_blocks[0], stride=1, use_checkpoint=use_checkpoint)
        self.layer2 = self._make_layer(64, 128, num_blocks[1], stride=2, use_checkpoint=use_checkpoint)
        self.layer3 = self._make_layer(128, 256, num_blocks[2], stride=2, use_checkpoint=use_checkpoint)
        self.layer4 = self._make_layer(256, 512, num_blocks[3], stride=2, use_checkpoint=use_checkpoint)
        
        # Global average pooling and final classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride, use_checkpoint):
        layers = []
        # First block may have stride > 1
        layers.append(ResidualBlock(in_channels, out_channels, stride, use_checkpoint))
        # Remaining blocks have stride = 1
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1, use_checkpoint))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Implement forward pass through all layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

def compare_memory_usage():
    """Compare memory usage with and without gradient checkpointing"""
    if not torch.cuda.is_available():
        print("CUDA not available - using CPU for demonstration")
        device = torch.device('cpu')
    else:
        device = torch.device('cuda')
    
    # Create two identical networks (with/without checkpointing)
    num_blocks = [2, 2, 2, 2]  # ResNet-18 style
    
    print("Comparing memory usage between regular ResNet and checkpointed ResNet...")
    
    # Test without checkpointing
    model_regular = CheckpointResNet(num_blocks, num_classes=10, use_checkpoint=False).to(device)
    
    # Test with checkpointing  
    model_checkpoint = CheckpointResNet(num_blocks, num_classes=10, use_checkpoint=True).to(device)
    
    # Create sample data
    batch_size = 8
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    target = torch.randint(0, 10, (batch_size,)).to(device)
    
    # Test regular model
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    
    model_regular.train()
    optimizer_regular = torch.optim.SGD(model_regular.parameters(), lr=0.01)
    
    # Forward and backward pass
    output_regular = model_regular(x)
    loss_regular = F.cross_entropy(output_regular, target)
    loss_regular.backward()
    
    if torch.cuda.is_available():
        memory_regular = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
        print(f"Regular ResNet peak memory: {memory_regular:.2f} MB")
    
    # Test checkpointed model
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
    
    model_checkpoint.train()
    optimizer_checkpoint = torch.optim.SGD(model_checkpoint.parameters(), lr=0.01)
    
    # Forward and backward pass
    output_checkpoint = model_checkpoint(x)
    loss_checkpoint = F.cross_entropy(output_checkpoint, target)
    loss_checkpoint.backward()
    
    if torch.cuda.is_available():
        memory_checkpoint = torch.cuda.max_memory_allocated(device) / 1024**2  # MB
        print(f"Checkpointed ResNet peak memory: {memory_checkpoint:.2f} MB")
        
        memory_savings = memory_regular - memory_checkpoint
        savings_percent = (memory_savings / memory_regular) * 100
        print(f"Memory savings: {memory_savings:.2f} MB ({savings_percent:.1f}%)")
    else:
        print("Memory comparison requires CUDA - both models ran successfully on CPU")

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