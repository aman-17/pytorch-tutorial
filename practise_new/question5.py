"""
Question 5: Multi-Scale Feature Pyramid Network (Hard)

Implement a Feature Pyramid Network (FPN) that combines features from multiple scales.
FPN is commonly used in object detection to handle objects of different sizes.

Key concepts:
1. Bottom-up pathway (backbone network)
2. Top-down pathway with lateral connections
3. Feature fusion at multiple scales
4. 1x1 convolutions for channel alignment

Requirements:
- Process features at 4 different scales
- Use lateral connections to combine high-level semantic information with low-level details
- Output feature maps at multiple scales

Example architecture:
Input -> Conv layers -> FPN -> Multi-scale outputs [P2, P3, P4, P5]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    """Basic residual block for backbone"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        # TODO: Implement basic residual block
        pass
    
    def forward(self, x):
        # TODO: Implement forward pass with residual connection
        pass

class Backbone(nn.Module):
    """Simple backbone network to extract multi-scale features"""
    def __init__(self):
        super(Backbone, self).__init__()
        # TODO: Create backbone layers that output features at 4 different scales
        # Typical channels: [64, 128, 256, 512]
        # Each stage should downsample by factor of 2
        pass
    
    def forward(self, x):
        # TODO: Return list of features [C2, C3, C4, C5] at different scales
        pass

class FeaturePyramidNetwork(nn.Module):
    def __init__(self, in_channels_list, out_channels=256):
        super(FeaturePyramidNetwork, self).__init__()
        # TODO: Initialize FPN components
        # 1. Lateral convolutions (1x1 conv to align channels)
        # 2. Output convolutions (3x3 conv for final features)
        pass
    
    def forward(self, features):
        """
        Args:
            features: List of feature maps [C2, C3, C4, C5] from backbone
        Returns:
            List of pyramid features [P2, P3, P4, P5]
        """
        # TODO: Implement FPN forward pass
        # 1. Start from highest level feature (C5)
        # 2. Apply lateral convolution to each feature
        # 3. Add upsampled higher-level feature to current lateral feature
        # 4. Apply output convolution to get final pyramid feature
        pass

class FPNNetwork(nn.Module):
    """Complete network with backbone + FPN"""
    def __init__(self, num_classes=10):
        super(FPNNetwork, self).__init__()
        # TODO: Initialize backbone and FPN
        # TODO: Add classifier heads for each pyramid level
        pass
    
    def forward(self, x):
        # TODO: Forward pass through backbone, FPN, and classifiers
        # Return predictions from multiple scales
        pass

# Test your implementation
if __name__ == "__main__":
    # Test FPN
    model = FPNNetwork(num_classes=10)
    x = torch.randn(2, 3, 224, 224)
    
    outputs = model(x)
    print("FPN outputs:")
    for i, output in enumerate(outputs):
        print(f"P{i+2} shape: {output.shape}")
    
    # Test that gradients flow properly
    loss = sum(output.sum() for output in outputs)
    loss.backward()
    print("Gradient check passed!")