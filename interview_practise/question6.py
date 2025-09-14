"""
Question 6: Differentiable Neural Architecture Search (NAS) Cell (Hard)

Implement a differentiable NAS cell that can learn optimal connections between operations.
The cell should use continuous relaxation of architecture search where each edge has
multiple possible operations and learns weights for each operation.

Key concepts:
1. Mixed operations with learnable alpha parameters
2. Softmax over operation choices
3. Gradient-based architecture optimization
4. Operations: Conv3x3, Conv5x5, MaxPool, AvgPool, Identity, Zero

Example: The cell learns which operations to use on each edge during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class MixedOp(nn.Module):
    """Mixed operation that combines multiple operations with learnable weights"""
    def __init__(self, channels, stride=1):
        super(MixedOp, self).__init__()
        self.ops = nn.ModuleList()
        # TODO: Define available operations
        # 1. 3x3 separable conv
        # 2. 5x5 separable conv  
        # 3. 3x3 max pooling
        # 4. 3x3 average pooling
        # 5. Identity (skip connection)
        # 6. Zero operation
        pass
    
    def forward(self, x, weights):
        """
        Args:
            x: input tensor
            weights: operation weights (softmax normalized)
        Returns:
            Weighted combination of all operations
        """
        # TODO: Apply each operation and combine with weights
        pass

class NASCell(nn.Module):
    """Differentiable NAS cell with learnable architecture"""
    def __init__(self, in_channels, out_channels, n_nodes=4):
        super(NASCell, self).__init__()
        # TODO: Initialize cell parameters
        # - n_nodes: number of intermediate nodes
        # - Each node can receive inputs from all previous nodes
        # - Architecture parameters (alpha) for each edge
        pass
    
    def forward(self, x):
        # TODO: Implement cell forward pass
        # 1. Process input through preprocessing layers
        # 2. For each node, collect inputs from all previous nodes
        # 3. Apply mixed operations with current architecture weights
        # 4. Combine outputs from final nodes
        pass
    
    def get_architecture_weights(self):
        """Return current architecture weights (alpha parameters)"""
        # TODO: Return softmax normalized architecture parameters
        pass
    
    def get_current_architecture(self):
        """Return the current best architecture (argmax of weights)"""
        # TODO: Return indices of strongest operations for each edge
        pass

class NASNetwork(nn.Module):
    """Complete network with multiple NAS cells"""
    def __init__(self, num_classes=10, num_cells=3):
        super(NASNetwork, self).__init__()
        # TODO: Stack multiple NAS cells
        # TODO: Add final classification layer
        pass
    
    def forward(self, x):
        # TODO: Forward pass through all cells
        pass
    
    def architecture_parameters(self):
        """Return all architecture parameters for optimization"""
        # TODO: Collect alpha parameters from all cells
        pass

# Test your implementation
if __name__ == "__main__":
    # Test NAS cell
    cell = NASCell(in_channels=16, out_channels=32, n_nodes=4)
    x = torch.randn(2, 16, 32, 32)
    output = cell(x)
    print(f"Cell output shape: {output.shape}")
    
    # Test architecture weights
    arch_weights = cell.get_architecture_weights()
    print(f"Architecture weights shape: {[w.shape for w in arch_weights]}")
    
    # Test full network
    model = NASNetwork(num_classes=10, num_cells=2)
    x = torch.randn(2, 3, 32, 32)
    output = model(x)
    print(f"Network output shape: {output.shape}")
    
    # Test architecture parameter collection
    arch_params = list(model.architecture_parameters())
    print(f"Total architecture parameters: {len(arch_params)}")