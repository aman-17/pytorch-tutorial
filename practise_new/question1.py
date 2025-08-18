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
        # Implement forward pass: f(x) = x * sigmoid(x)
        sigmoid = torch.sigmoid(input)
        output = input * sigmoid
        
        # Save tensors needed for backward pass
        ctx.save_for_backward(input, sigmoid)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        # Implement backward pass
        # f'(x) = σ(x) + x * σ(x) * (1 - σ(x))
        input, sigmoid = ctx.saved_tensors
        
        # Compute the derivative
        derivative = sigmoid + input * sigmoid * (1 - sigmoid)
        
        # Apply chain rule
        return grad_output * derivative

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