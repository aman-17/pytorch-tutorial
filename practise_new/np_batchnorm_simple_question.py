"""
Question: Simple Batch Normalization (Easy)

Implement a simplified version of Batch Normalization for understanding the core concepts.
BatchNorm normalizes inputs to have zero mean and unit variance, then scales and shifts.

Forward pass:
1. Compute mean and variance across batch
2. Normalize: x_norm = (x - mean) / sqrt(variance + eps)
3. Scale and shift: output = gamma * x_norm + beta

This question focuses on the basic math, not the full complexity of real BatchNorm.

Your task: Implement forward and backward passes for batch normalization.
"""

import numpy as np

def batchnorm_forward(x, gamma, beta, eps=1e-5):
    """
    Simplified Batch Normalization forward pass.
    
    Args:
        x: Input of shape (batch_size, num_features)
        gamma: Scale parameter, shape (num_features,)
        beta: Shift parameter, shape (num_features,)
        eps: Small value for numerical stability
    
    Returns:
        output: Normalized, scaled and shifted data
        cache: Values needed for backward pass
    """
    # TODO: Implement batch normalization forward pass
    # Steps:
    # 1. Compute mean across batch dimension (axis=0)
    # 2. Compute variance across batch dimension (axis=0)
    # 3. Normalize: (x - mean) / sqrt(var + eps)
    # 4. Scale and shift: gamma * x_normalized + beta
    # 5. Cache all intermediate values for backward pass
    
    output = None  # Replace with your implementation
    cache = None  # Store mean, var, x_normalized, etc.
    
    return output, cache

def batchnorm_backward(dout, cache):
    """
    Simplified Batch Normalization backward pass.
    
    Args:
        dout: Gradient from upstream, same shape as x
        cache: Values saved from forward pass
    
    Returns:
        dx: Gradient w.r.t input x
        dgamma: Gradient w.r.t scale parameter
        dbeta: Gradient w.r.t shift parameter
    """
    # TODO: Implement batch normalization backward pass
    # This is more complex! Key steps:
    # 1. dgamma = sum(dout * x_normalized) across batch
    # 2. dbeta = sum(dout) across batch
    # 3. dx requires several steps due to normalization
    
    dx = None  # Replace with your implementation
    dgamma = None  # Replace with your implementation  
    dbeta = None  # Replace with your implementation
    
    return dx, dgamma, dbeta

def test_batchnorm_forward():
    """Test batch normalization forward pass."""
    print("=== Testing BatchNorm Forward Pass ===")
    
    # Simple test case
    batch_size = 4
    num_features = 3
    
    # Create input with different scales per feature
    x = np.array([[1, 100, 0.1],
                  [2, 200, 0.2],
                  [3, 300, 0.3],
                  [4, 400, 0.4]])
    
    # Initialize parameters
    gamma = np.ones(num_features)
    beta = np.zeros(num_features)
    
    print(f"Input shape: {x.shape}")
    print(f"Input:\n{x}")
    
    output, cache = batchnorm_forward(x, gamma, beta)
    
    if output is not None:
        print(f"\nOutput shape: {output.shape}")
        print(f"Output:\n{output}")
        
        # Check normalization properties
        print(f"\nOutput statistics per feature:")
        print(f"Means: {np.mean(output, axis=0)} (should be ~0)")
        print(f"Stds: {np.std(output, axis=0)} (should be ~1)")
        
        # Test with different gamma and beta
        print("\n=== Testing with gamma=2, beta=1 ===")
        gamma = np.array([2.0, 2.0, 2.0])
        beta = np.array([1.0, 1.0, 1.0])
        
        output2, _ = batchnorm_forward(x, gamma, beta)
        if output2 is not None:
            print(f"Output:\n{output2}")
            print(f"Means: {np.mean(output2, axis=0)} (should be ~1)")
            print(f"Stds: {np.std(output2, axis=0)} (should be ~2)")
    else:
        print("Forward pass not implemented yet")

def test_batchnorm_backward():
    """Test batch normalization backward pass."""
    print("\n=== Testing BatchNorm Backward Pass ===")
    
    # Simple setup
    batch_size = 2
    num_features = 3
    
    np.random.seed(42)
    x = np.random.randn(batch_size, num_features)
    gamma = np.ones(num_features)
    beta = np.zeros(num_features)
    
    output, cache = batchnorm_forward(x, gamma, beta)
    
    if output is not None:
        # Create upstream gradient
        dout = np.ones_like(output)
        
        dx, dgamma, dbeta = batchnorm_backward(dout, cache)
        
        if dx is not None and dgamma is not None and dbeta is not None:
            print(f"dx shape: {dx.shape} (should match input shape)")
            print(f"dgamma shape: {dgamma.shape} (should be {gamma.shape})")
            print(f"dbeta shape: {dbeta.shape} (should be {beta.shape})")
            
            # Numerical gradient check for gamma
            print("\n=== Numerical Gradient Check (gamma[0]) ===")
            eps = 1e-5
            
            gamma_plus = gamma.copy()
            gamma_plus[0] += eps
            out_plus, _ = batchnorm_forward(x, gamma_plus, beta)
            
            gamma_minus = gamma.copy()
            gamma_minus[0] -= eps
            out_minus, _ = batchnorm_forward(x, gamma_minus, beta)
            
            numerical_grad = np.sum((out_plus - out_minus) * dout) / (2 * eps)
            analytical_grad = dgamma[0]
            
            print(f"Numerical gradient: {numerical_grad:.6f}")
            print(f"Analytical gradient: {analytical_grad:.6f}")
            
            error = abs(numerical_grad - analytical_grad)
            print(f"Error: {error:.2e} (should be < 1e-5)")
        else:
            print("Backward pass not implemented yet")

def test_gradient_flow():
    """Test that gradients flow correctly through batch norm."""
    print("\n=== Testing Gradient Flow ===")
    
    batch_size = 8
    num_features = 5
    
    np.random.seed(123)
    x = np.random.randn(batch_size, num_features) * 10 + 5  # Different scale
    gamma = np.ones(num_features) * 0.5
    beta = np.ones(num_features) * 0.1
    
    output, cache = batchnorm_forward(x, gamma, beta)
    
    if output is not None:
        # Test with different upstream gradients
        test_cases = [
            ("Uniform gradient", np.ones_like(output)),
            ("Random gradient", np.random.randn(*output.shape)),
            ("Zero gradient", np.zeros_like(output))
        ]
        
        for name, dout in test_cases:
            dx, dgamma, dbeta = batchnorm_backward(dout, cache)
            
            if dx is not None:
                print(f"\n{name}:")
                print(f"  Mean |dx|: {np.mean(np.abs(dx)):.6f}")
                print(f"  Mean |dgamma|: {np.mean(np.abs(dgamma)):.6f}")
                print(f"  Mean |dbeta|: {np.mean(np.abs(dbeta)):.6f}")

if __name__ == "__main__":
    test_batchnorm_forward()
    test_batchnorm_backward()
    test_gradient_flow()
    
    print("\n=== Key Concepts ===")
    print("1. BatchNorm normalizes each feature across the batch")
    print("2. Learnable parameters (gamma, beta) allow the network to undo normalization if needed")
    print("3. Helps with training stability and allows higher learning rates")
    print("4. The backward pass is complex due to interdependencies in normalization")
    print("5. In practice, running statistics are kept for inference")