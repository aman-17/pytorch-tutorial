"""
Question: Linear Layer Forward and Backward Pass (Easy)

Implement a simple linear (fully connected) layer with forward and backward propagation.
A linear layer performs: y = x @ W + b
Where @ denotes matrix multiplication.

This is the fundamental building block of neural networks!

Your task: Implement forward and backward passes for a linear layer.

Example:
Input shape: (2, 3) - 2 samples, 3 features
Weight shape: (3, 4) - transforms 3 features to 4 features  
Bias shape: (4,)
Output shape: (2, 4) - 2 samples, 4 output features
"""

import numpy as np

class LinearLayer:
    def __init__(self, input_dim, output_dim):
        """
        Initialize linear layer with random weights.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output features
        """
        # Initialize weights and bias
        # Using small random values for weights
        self.W = np.random.randn(input_dim, output_dim) * 0.01
        self.b = np.zeros(output_dim)
        
        # Cache for backward pass
        self.cache = {}
        
    def forward(self, x):
        """
        Forward pass: y = x @ W + b
        
        Args:
            x: Input of shape (batch_size, input_dim)
            
        Returns:
            output: Shape (batch_size, output_dim)
        """
        # TODO: Implement forward pass
        # Hint: Use np.dot or @ operator for matrix multiplication
        # Don't forget to add the bias!
        # Save x in self.cache for backward pass
        self.cache['x'] = x
        self.cache['b'] = self.b

        output = np.matmul(x, self.W) + self.b  # Replace with your implementation

        return output
    
    def backward(self, dout):
        """
        Backward pass: compute gradients.
        
        Args:
            dout: Gradient from upstream, shape (batch_size, output_dim)
            
        Returns:
            dx: Gradient w.r.t input, shape (batch_size, input_dim)
            dW: Gradient w.r.t weights, shape (input_dim, output_dim)
            db: Gradient w.r.t bias, shape (output_dim,)
        """
        # TODO: Implement backward pass
        # Hints:
        # 1. dx = dout @ W.T (gradient w.r.t input)
        # 2. dW = x.T @ dout (gradient w.r.t weights)
        # 3. db = sum of dout along batch dimension
        
        dx = np.matmul(dout, np.swapaxes(self.W, -2, -1))  # Replace with your implementation
        dW = np.matmul(np.swapaxes(self.cache['x'], -2, -1), dout)  # Replace with your implementation
        db = np.sum(dout, axis=-1, keepdims=True)  # Replace with your implementation
        
        return dx, dW, db

def test_linear_layer():
    """Test your linear layer implementation."""
    print("=== Testing Linear Layer ===")
    
    # Create a simple linear layer
    batch_size = 2
    input_dim = 3
    output_dim = 4
    
    layer = LinearLayer(input_dim, output_dim)
    
    # Create sample input
    x = np.array([[1.0, 2.0, 3.0],
                  [4.0, 5.0, 6.0]])
    
    print(f"Input shape: {x.shape}")
    print(f"Weight shape: {layer.W.shape}")
    print(f"Bias shape: {layer.b.shape}")
    
    # Test forward pass
    print("\n=== Forward Pass ===")
    output = layer.forward(x)
    
    if output is not None:
        print(f"Output shape: {output.shape}")
        print(f"Output:\n{output}")
        
        # Test backward pass
        print("\n=== Backward Pass ===")
        dout = np.ones((batch_size, output_dim))
        dx, dW, db = layer.backward(dout)
        
        if dx is not None and dW is not None and db is not None:
            print(f"dx shape: {dx.shape} (should be {x.shape})")
            print(f"dW shape: {dW.shape} (should be {layer.W.shape})")
            print(f"db shape: {db.shape} (should be {layer.b.shape})")
            
            # Numerical gradient check for weights
            print("\n=== Numerical Gradient Check (First Weight) ===")
            epsilon = 1e-5
            
            # Check gradient for W[0,0]
            layer.W[0, 0] += epsilon
            out_plus = layer.forward(x)
            
            layer.W[0, 0] -= 2 * epsilon
            out_minus = layer.forward(x)
            
            layer.W[0, 0] += epsilon  # Restore original value
            
            numerical_grad = np.sum((out_plus - out_minus) * dout) / (2 * epsilon)
            analytical_grad = dW[0, 0]
            
            print(f"Numerical gradient for W[0,0]: {numerical_grad:.6f}")
            print(f"Analytical gradient for W[0,0]: {analytical_grad:.6f}")
            
            error = abs(numerical_grad - analytical_grad)
            print(f"Error: {error:.2e} (should be < 1e-7)")
        else:
            print("Backward pass not implemented yet")
    else:
        print("Forward pass not implemented yet")

def test_batch_processing():
    """Test that your implementation handles batches correctly."""
    print("\n=== Testing Batch Processing ===")
    
    layer = LinearLayer(5, 3)
    
    # Test with different batch sizes
    for batch_size in [1, 4, 10]:
        x = np.random.randn(batch_size, 5)
        output = layer.forward(x)
        
        if output is not None:
            print(f"Batch size {batch_size}: Input {x.shape} -> Output {output.shape}")
            
            # Check backward pass
            dout = np.random.randn(batch_size, 3)
            dx, dW, db = layer.backward(dout)
            
            if dx is not None:
                assert dx.shape == x.shape, f"dx shape mismatch: {dx.shape} != {x.shape}"
                assert dW.shape == layer.W.shape, f"dW shape mismatch: {dW.shape} != {layer.W.shape}"
                assert db.shape == layer.b.shape, f"db shape mismatch: {db.shape} != {layer.b.shape}"
                print(f"  ✓ Gradient shapes correct")

if __name__ == "__main__":
    test_linear_layer()
    test_batch_processing()
    
    print("\n=== Key Concepts ===")
    print("1. Forward: y = x @ W + b")
    print("2. Backward (chain rule):")
    print("   - dx = dout @ W.T")
    print("   - dW = x.T @ dout")
    print("   - db = sum(dout, axis=0)")
    print("3. Shapes must be consistent throughout!")