"""
Question 4: Advanced Batch Normalization Implementation (Medium-Hard)

Implement a custom Batch Normalization layer from scratch that handles both training and
evaluation modes correctly. Your implementation should match PyTorch's BatchNorm1d behavior.

Key concepts to implement:
1. Running statistics (mean and variance) with momentum
2. Training vs evaluation mode differences
3. Learnable parameters (gamma and beta)
4. Numerical stability with epsilon

Example:
Input: tensor([[1., 2., 3.], [4., 5., 6.]])  # shape: (batch_size, features)
After normalization, each feature should have approximately mean=0, std=1
"""

import torch
import torch.nn as nn

class CustomBatchNorm1d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True):
        super(CustomBatchNorm1d, self).__init__()
        # TODO: Initialize parameters
        # - num_features: number of features/channels
        # - eps: small value for numerical stability
        # - momentum: momentum for running statistics
        # - affine: whether to use learnable parameters
        
        # TODO: Create learnable parameters (gamma, beta) if affine=True
        
        # TODO: Create buffers for running statistics (running_mean, running_var)
        
        # TODO: Store other parameters
        pass
    
    def forward(self, input):
        # TODO: Implement forward pass
        # Training mode:
        # 1. Calculate batch statistics (mean, variance)
        # 2. Update running statistics with momentum
        # 3. Normalize using batch statistics
        
        # Evaluation mode:
        # 1. Use running statistics for normalization
        
        # Both modes:
        # 1. Apply affine transformation if enabled
        pass
    
    def extra_repr(self):
        # TODO: Return string representation of layer parameters
        pass

def test_custom_batch_norm():
    """Test custom implementation against PyTorch's BatchNorm1d"""
    # TODO: Create both custom and PyTorch BatchNorm layers
    # TODO: Test with same input and compare outputs
    # TODO: Test training vs eval mode behavior
    # TODO: Check that running statistics are updated correctly
    pass

# Test your implementation
if __name__ == "__main__":
    # Basic test
    bn = CustomBatchNorm1d(3)
    x = torch.randn(4, 3, requires_grad=True)
    
    # Training mode
    bn.train()
    output_train = bn(x)
    print(f"Training output shape: {output_train.shape}")
    print(f"Training output mean: {output_train.mean(dim=0)}")
    print(f"Training output std: {output_train.std(dim=0)}")
    
    # Evaluation mode
    bn.eval()
    output_eval = bn(x)
    print(f"Eval output shape: {output_eval.shape}")
    
    # Run comprehensive test
    test_custom_batch_norm()