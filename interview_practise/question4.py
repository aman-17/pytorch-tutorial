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
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = True
        
        # Create learnable parameters (gamma, beta) if affine=True
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        # Create buffers for running statistics (running_mean, running_var)
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
    
    def forward(self, input):
        # Calculate exponential average factor
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum
        
        if self.training and self.track_running_stats:
            # Update num_batches_tracked
            if self.num_batches_tracked is not None:
                self.num_batches_tracked = self.num_batches_tracked + 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        
        if self.training:
            # Calculate batch statistics (mean, variance)
            batch_mean = input.mean(dim=0, keepdim=False)
            batch_var = input.var(dim=0, keepdim=False, unbiased=False)
            
            # Update running statistics with momentum
            if self.track_running_stats:
                with torch.no_grad():
                    self.running_mean = exponential_average_factor * batch_mean + \
                                      (1 - exponential_average_factor) * self.running_mean
                    # Update running_var with unbiased estimation
                    n = input.numel() / input.size(1)
                    self.running_var = exponential_average_factor * batch_var * n / (n - 1) + \
                                     (1 - exponential_average_factor) * self.running_var
            
            # Use batch statistics for normalization
            mean = batch_mean
            var = batch_var
        else:
            # Use running statistics for normalization
            mean = self.running_mean
            var = self.running_var
        
        # Normalize the input
        input_normalized = (input - mean) / torch.sqrt(var + self.eps)
        
        # Apply affine transformation if enabled
        if self.affine:
            input_normalized = input_normalized * self.weight + self.bias
        
        return input_normalized
    
    def extra_repr(self):
        return f'{self.num_features}, eps={self.eps}, momentum={self.momentum}, affine={self.affine}, track_running_stats={self.track_running_stats}'

def test_custom_batch_norm():
    """Test custom implementation against PyTorch's BatchNorm1d"""
    print("Testing Custom BatchNorm1d implementation...")
    
    # Create both custom and PyTorch BatchNorm layers
    num_features = 3
    custom_bn = CustomBatchNorm1d(num_features, momentum=0.1)
    pytorch_bn = nn.BatchNorm1d(num_features, momentum=0.1)
    
    # Initialize with same parameters
    with torch.no_grad():
        if custom_bn.weight is not None:
            pytorch_bn.weight.copy_(custom_bn.weight)
        if custom_bn.bias is not None:
            pytorch_bn.bias.copy_(custom_bn.bias)
        pytorch_bn.running_mean.copy_(custom_bn.running_mean)
        pytorch_bn.running_var.copy_(custom_bn.running_var)
    
    # Test with same input
    x = torch.randn(4, 3, requires_grad=True)
    
    # Test training mode
    custom_bn.train()
    pytorch_bn.train()
    
    custom_output = custom_bn(x)
    pytorch_output = pytorch_bn(x)
    
    print(f"Training mode output difference: {torch.max(torch.abs(custom_output - pytorch_output)).item():.6f}")
    print(f"Custom running mean: {custom_bn.running_mean}")
    print(f"PyTorch running mean: {pytorch_bn.running_mean}")
    print(f"Running mean difference: {torch.max(torch.abs(custom_bn.running_mean - pytorch_bn.running_mean)).item():.6f}")
    
    # Test evaluation mode
    custom_bn.eval()
    pytorch_bn.eval()
    
    custom_eval_output = custom_bn(x)
    pytorch_eval_output = pytorch_bn(x)
    
    print(f"Eval mode output difference: {torch.max(torch.abs(custom_eval_output - pytorch_eval_output)).item():.6f}")
    
    # Test that outputs are normalized (mean ~0, std ~1)
    print(f"Custom output mean per feature: {custom_output.mean(dim=0)}")
    print(f"Custom output std per feature: {custom_output.std(dim=0)}")
    
    print("Test completed!")

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