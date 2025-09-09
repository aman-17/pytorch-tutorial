"""
Question: ReLU and Softmax Activation Functions (Easy)

Implement two important activation functions:
1. ReLU (Rectified Linear Unit): f(x) = max(0, x)
2. Softmax: Converts logits to probabilities

ReLU is the most popular activation in hidden layers.
Softmax is used in the output layer for multi-class classification.

Your task: Implement forward and backward passes for both.
"""

import numpy as np

def relu_forward(x):
    """
    ReLU activation: f(x) = max(0, x)
    
    Args:
        x: Input array of any shape
    
    Returns:
        output: ReLU of x, same shape as input
        cache: Values needed for backward pass
    """
    # TODO: Implement ReLU forward pass
    # Hint: Use np.maximum(0, x) or equivalent
    # Cache the input for backward pass
    
    output = np.maximum(0, x)  # Replace with your implementation
    cache = output  # Store what you need for backward
    
    return output, cache

def relu_backward(dout, cache):
    """
    ReLU backward pass.
    
    Args:
        dout: Gradient from upstream
        cache: Values saved from forward pass
    
    Returns:
        dx: Gradient with respect to input
    """
    # TODO: Implement ReLU backward pass
    # Hint: Gradient is 1 where x > 0, else 0
    # This means gradient passes through where input was positive

    dx = dout * (1 if cache.any() > 0 else 0)
    return dx

def softmax_forward(x):
    """
    Softmax activation: converts logits to probabilities.
    
    Args:
        x: Input array of shape (batch_size, num_classes)
    
    Returns:
        output: Probabilities, same shape as input, sum to 1 along last axis
        cache: Values needed for backward pass
    """
    # TODO: Implement softmax forward pass
    # Steps:
    # 1. Subtract max for numerical stability: x - np.max(x, axis=-1, keepdims=True)
    # 2. Compute exp: np.exp(shifted_x)
    # 3. Normalize: exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    output = x - np.max(x, axis=-1, keepdims=True)
    output = np.exp(output) / np.sum(np.exp(output), axis=-1, keepdims=True)
    cache = output  # Store what you need for backward

    return output, cache

def softmax_backward(dout, cache):
    """
    Softmax backward pass (simplified version).
    
    Args:
        dout: Gradient from upstream, shape (batch_size, num_classes)
        cache: Softmax output from forward pass
    
    Returns:
        dx: Gradient with respect to input
    """
    # TODO: Implement softmax backward pass
    # For softmax output p and upstream gradient dout:
    # dx = p * (dout - sum(dout * p))
    # This is the simplified formula when used with cross-entropy loss

    dx = cache * (dout - np.sum(dout * cache, axis=-1, keepdims=True))

    return dx

def test_relu():
    """Test ReLU implementation."""
    print("=== Testing ReLU ===")
    x = np.array([[-2, -1, 0, 1, 2],
                  [3, -4, 5, -6, 7]])
    
    output, cache = relu_forward(x)
    
    if output is not None:
        print(f"Input:\n{x}")
        print(f"ReLU output:\n{output}")
        print(f"Expected:\n[[0, 0, 0, 1, 2],\n [3, 0, 5, 0, 7]]")
        
        # Test backward
        print("\n=== ReLU Backward ===")
        dout = np.ones_like(output)
        dx = relu_backward(dout, cache)
        
        if dx is not None:
            print(f"Gradient:\n{dx}")
            print(f"Expected (1 where input > 0, else 0):\n[[0, 0, 0, 1, 1],\n [1, 0, 1, 0, 1]]")
    else:
        print("ReLU forward not implemented yet")

def test_softmax():
    """Test Softmax implementation."""
    print("\n=== Testing Softmax ===")
    # Logits for 3 samples, 4 classes
    x = np.array([[2.0, 1.0, 0.1, 0.0],
                  [1.0, 2.0, 3.0, 4.0],
                  [1.0, 1.0, 1.0, 1.0]])
    
    output, cache = softmax_forward(x)
    
    if output is not None:
        print(f"Input logits:\n{x}")
        print(f"Softmax output:\n{output}")
        print(f"Sum of probabilities per sample: {np.sum(output, axis=1)}")
        print("(Should all be 1.0)")
        
        # Check properties
        assert np.allclose(np.sum(output, axis=1), 1.0), "Probabilities don't sum to 1!"
        assert np.all(output >= 0) and np.all(output <= 1), "Output not in [0,1]!"
        print("✓ Softmax properties verified")
        
        # Test backward
        print("\n=== Softmax Backward ===")
        dout = np.array([[1, 0, 0, 0],  # Gradient from cross-entropy with label 0
                         [0, 0, 0, 1],  # Gradient from cross-entropy with label 3
                         [0.25, 0.25, 0.25, 0.25]])  # Uniform gradient
        
        dx = softmax_backward(dout, cache)
        
        if dx is not None:
            print(f"Upstream gradient:\n{dout}")
            print(f"Gradient w.r.t input:\n{dx}")
            print("Note: Rows should sum to ~0 due to softmax properties")
            print(f"Row sums: {np.sum(dx, axis=1)}")
    else:
        print("Softmax forward not implemented yet")

def test_numerical_gradient():
    """Numerical gradient check for both functions."""
    print("\n=== Numerical Gradient Check ===")
    
    # Test ReLU
    print("ReLU gradient check:")
    x = np.array([[1.5, -2.0, 0.5]])
    out, cache = relu_forward(x)
    
    if out is not None:
        dout = np.ones_like(out)
        dx_analytical = relu_backward(dout, cache)
        
        if dx_analytical is not None:
            epsilon = 1e-5
            dx_numerical = np.zeros_like(x)
            
            for i in range(x.shape[1]):
                x_plus = x.copy()
                x_plus[0, i] += epsilon
                out_plus, _ = relu_forward(x_plus)
                
                x_minus = x.copy()
                x_minus[0, i] -= epsilon
                out_minus, _ = relu_forward(x_minus)
                
                dx_numerical[0, i] = np.sum((out_plus - out_minus) * dout) / (2 * epsilon)
            
            print(f"Analytical: {dx_analytical}")
            print(f"Numerical:  {dx_numerical}")
            print(f"Max error: {np.max(np.abs(dx_analytical - dx_numerical)):.2e}")

if __name__ == "__main__":
    test_relu()
    test_softmax()
    test_numerical_gradient()
    
    print("\n=== Key Points ===")
    print("1. ReLU: Simple but effective, gradient is binary (0 or 1)")
    print("2. Softmax: Converts any real values to valid probabilities")
    print("3. Always subtract max in softmax for numerical stability!")
    print("4. Softmax gradient has special structure when used with cross-entropy")