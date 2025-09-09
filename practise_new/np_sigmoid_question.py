"""
Question: Sigmoid Activation Forward and Backward Pass (Easy)

Implement the sigmoid activation function and its gradient using numpy.
The sigmoid function is one of the most fundamental activation functions in neural networks.

Sigmoid: σ(x) = 1 / (1 + e^(-x))
Gradient: σ'(x) = σ(x) * (1 - σ(x))

Your task: Implement the forward and backward functions below.

Example:
Input: [0.0, 1.0, -1.0, 2.0]
Sigmoid output: [0.5, 0.731, 0.269, 0.881]
Gradient at those points: [0.25, 0.196, 0.196, 0.105]
"""

import numpy as np

def sigmoid_forward(x):
    """
    Compute sigmoid activation function.
    
    Args:
        x: Input array of any shape
    
    Returns:
        output: Sigmoid of x, same shape as input
        cache: Values needed for backward pass
    """
    # TODO: Implement sigmoid forward pass
    # Hint: output = 1 / (1 + np.exp(-x))
    # Store the output in cache for backward pass
    output = 1 / (1 + np.exp(-x))
    cache = output  # Store the output for use in backward pass

    return np.round(output, decimals=3), np.round(cache, decimals=3)

def sigmoid_backward(dout, cache):
    """
    Compute gradient of sigmoid function.
    
    Args:
        dout: Gradient from upstream, same shape as sigmoid output
        cache: Values saved from forward pass
    
    Returns:
        dx: Gradient with respect to input x
    """
    # TODO: Implement sigmoid backward pass
    # Hint: If sigmoid_output is cached, gradient = sigmoid_output * (1 - sigmoid_output)
    # Then apply chain rule: dx = dout * local_gradient

    dx = dout * cache * (1 - cache)
    return np.round(dx, decimals=3)

def test_sigmoid():
    """Test your sigmoid implementation."""
    print("=== Testing Sigmoid Forward Pass ===")
    x = np.array([0.0, 1.0, -1.0, 2.0, -2.0])
    
    output, cache = sigmoid_forward(x)
    
    if output is not None:
        print(f"Input: {x}")
        print(f"Sigmoid output: {output}")
        print(f"Expected approximately: [0.5, 0.731, 0.269, 0.881, 0.119]")
        
        # Test backward pass
        print("\n=== Testing Sigmoid Backward Pass ===")
        dout = np.ones_like(output)  # Gradient of 1 from upstream
        dx = sigmoid_backward(dout, cache)
        
        if dx is not None:
            print(f"Gradient: {dx}")
            print(f"Expected approximately: [0.25, 0.196, 0.196, 0.105, 0.105]")
            
            # Numerical gradient check
            print("\n=== Numerical Gradient Check ===")
            epsilon = 1e-5
            numerical_grad = np.zeros_like(x)
            
            for i in range(len(x)):
                x_plus = x.copy()
                x_plus[i] += epsilon
                out_plus, _ = sigmoid_forward(x_plus)
                
                x_minus = x.copy()
                x_minus[i] -= epsilon
                out_minus, _ = sigmoid_forward(x_minus)
                
                numerical_grad[i] = np.sum((out_plus - out_minus) * dout) / (2 * epsilon)
            
            print(f"Numerical gradient: {numerical_grad}")
            print(f"Analytical gradient: {dx}")
            
            error = np.max(np.abs(numerical_grad - dx))
            print(f"Max error: {error:.2e} (should be < 1e-7)")
    else:
        print("Forward pass not implemented yet")

if __name__ == "__main__":
    test_sigmoid()
    
    print("\n=== Additional Test: 2D Array ===")
    x_2d = np.array([[1, -1], [0, 2]])
    output_2d, cache_2d = sigmoid_forward(x_2d)
    
    if output_2d is not None:
        print(f"Input shape: {x_2d.shape}")
        print(f"Output shape: {output_2d.shape}")
        print(f"Output:\n{output_2d}")