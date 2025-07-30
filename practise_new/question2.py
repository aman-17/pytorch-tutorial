"""
Question 2: Dynamic Neural Network with Variable Depth (Hard)

Implement a neural network that can dynamically change its depth during forward pass based on
the input data. The network should have a "confidence threshold" - if the prediction confidence
is above the threshold, it exits early; otherwise, it continues through more layers.

Requirements:
1. Start with 3 layers minimum, can go up to 6 layers maximum
2. After each layer (starting from layer 3), check if max probability > threshold
3. If yes, return prediction; if no, continue to next layer
4. Track which layer made the final prediction

Example usage should produce different exit layers for different inputs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DynamicDepthNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, confidence_threshold=0.8):
        super(DynamicDepthNetwork, self).__init__()
        # TODO: Initialize layers (6 layers total)
        # TODO: Store confidence_threshold
        pass
    
    def forward(self, x, return_exit_layer=False):
        # TODO: Implement dynamic forward pass
        # 1. Pass through first 3 layers (mandatory)
        # 2. For layers 4-6, check confidence after each layer
        # 3. Return prediction and optionally the exit layer number
        pass
    
    def get_confidence(self, logits):
        # TODO: Calculate confidence as max probability after softmax
        pass

# Test your implementation
if __name__ == "__main__":
    # Create network
    net = DynamicDepthNetwork(input_size=784, hidden_size=128, num_classes=10, confidence_threshold=0.9)
    
    # Test with different inputs
    batch_size = 4
    x1 = torch.randn(batch_size, 784)  # Random input
    x2 = torch.ones(batch_size, 784) * 0.5  # More predictable input
    
    with torch.no_grad():
        output1, exit_layer1 = net(x1, return_exit_layer=True)
        output2, exit_layer2 = net(x2, return_exit_layer=True)
        
        print(f"Input 1 - Exit layer: {exit_layer1}, Max confidence: {torch.max(F.softmax(output1, dim=1)):.3f}")
        print(f"Input 2 - Exit layer: {exit_layer2}, Max confidence: {torch.max(F.softmax(output2, dim=1)):.3f}")