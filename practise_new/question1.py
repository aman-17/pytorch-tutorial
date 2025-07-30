"""
Question 1: Custom Autograd Function (Medium-Hard)

Implement a custom autograd function for the Swish activation function: f(x) = x * sigmoid(x).
You need to implement both the forward pass and backward pass (gradient computation).

The Swish function is: f(x) = x * σ(x) where σ(x) = 1/(1 + e^(-x))
The derivative is: f'(x) = σ(x) + x * σ(x) * (1 - σ(x))

Example:
Input: tensor([1.0, 2.0, -1.0, 0.0])
Expected Output: tensor([0.7311, 1.7616, -0.2689, 0.0000])

Your task: Complete the SwishFunction class below.
"""

import torch
import torch.nn as nn
from torch.autograd import Function

class SwishFunction(Function):
    @staticmethod
    def forward(ctx, input):
        # TODO: Implement forward pass
        # Save any tensors needed for backward pass using ctx.save_for_backward()
        pass
    
    @staticmethod
    def backward(ctx, grad_output):
        # TODO: Implement backward pass
        # Retrieve saved tensors using ctx.saved_tensors
        pass

class Swish(nn.Module):
    def forward(self, x):
        return SwishFunction.apply(x)

# Test your implementation
if __name__ == "__main__":
    x = torch.tensor([1.0, 2.0, -1.0, 0.0], requires_grad=True)
    swish = Swish()
    y = swish(x)
    print(f"Output: {y}")
    
    # Test gradient
    loss = y.sum()
    loss.backward()
    print(f"Gradients: {x.grad}")