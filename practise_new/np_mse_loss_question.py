"""
Question: Mean Squared Error Loss (Easy)

Implement the Mean Squared Error (MSE) loss function and its gradient.
MSE is one of the most common loss functions for regression tasks.

MSE = (1/N) * Σ(predictions - targets)²

Your task: Implement forward pass (compute loss) and backward pass (compute gradients).

Example:
Predictions: [2.5, 0.0, 2.0, 8.0]
Targets:     [3.0, -0.5, 2.0, 7.0]  
MSE Loss: 0.375
"""

import numpy as np

def mse_loss_forward(predictions, targets):
    """
    Compute Mean Squared Error loss.
    
    Args:
        predictions: Model predictions, shape (batch_size, ...)
        targets: Ground truth values, same shape as predictions
    
    Returns:
        loss: Scalar MSE loss value
        cache: Values needed for backward pass
    """
    # TODO: Implement MSE forward pass
    # Steps:
    # 1. Compute squared differences: (predictions - targets)²
    # 2. Take mean over all elements
    # Cache predictions and targets for backward

    loss = np.mean(np.square(predictions - targets))  # Replace with your implementation
    cache = predictions - targets  # Store what you need for backward

    return loss, cache

def mse_loss_backward(cache):
    """
    Compute gradient of MSE loss with respect to predictions.
    
    Args:
        cache: Values saved from forward pass
    
    Returns:
        dpredictions: Gradient of loss w.r.t predictions
    """
    # TODO: Implement MSE backward pass
    # The gradient is: (2/N) * (predictions - targets)
    # where N is the total number of elements
    N = cache.size
    dpredictions = (2 / N) * cache  # Replace with your implementation

    return dpredictions

def test_mse_simple():
    """Test MSE with simple 1D arrays."""
    print("=== Testing MSE Loss - Simple Case ===")
    
    predictions = np.array([2.5, 0.0, 2.0, 8.0])
    targets = np.array([3.0, -0.5, 2.0, 7.0])
    
    print(f"Predictions: {predictions}")
    print(f"Targets: {targets}")
    
    loss, cache = mse_loss_forward(predictions, targets)
    
    if loss is not None:
        print(f"MSE Loss: {loss:.4f}")
        print(f"Expected: 0.375")
        
        # Manual calculation for verification
        differences = predictions - targets
        print(f"Differences: {differences}")
        print(f"Squared differences: {differences**2}")
        print(f"Mean of squared differences: {np.mean(differences**2):.4f}")
        
        # Test backward pass
        print("\n=== Testing Backward Pass ===")
        dpredictions = mse_loss_backward(cache)
        
        if dpredictions is not None:
            print(f"Gradient shape: {dpredictions.shape}")
            print(f"Gradient: {dpredictions}")
            
            expected_grad = 2 * (predictions - targets) / len(predictions)
            print(f"Expected gradient: {expected_grad}")
            
            # Check gradient magnitude
            print(f"\nGradient properties:")
            print(f"- Points toward targets (negative where pred > target)")
            print(f"- Proportional to prediction error")
    else:
        print("Forward pass not implemented yet")

def test_mse_batched():
    """Test MSE with batched 2D data."""
    print("\n=== Testing MSE Loss - Batched Data ===")
    
    batch_size = 3
    output_dim = 4
    
    # Random predictions and targets
    np.random.seed(42)
    predictions = np.random.randn(batch_size, output_dim)
    targets = np.random.randn(batch_size, output_dim)
    
    print(f"Predictions shape: {predictions.shape}")
    print(f"Targets shape: {targets.shape}")
    
    loss, cache = mse_loss_forward(predictions, targets)
    
    if loss is not None:
        print(f"MSE Loss: {loss:.4f}")
        
        # Test backward pass
        dpredictions = mse_loss_backward(cache)
        
        if dpredictions is not None:
            print(f"Gradient shape: {dpredictions.shape}")
            assert dpredictions.shape == predictions.shape, "Gradient shape mismatch!"
            print("✓ Gradient shape matches predictions")
            
            # Numerical gradient check
            print("\n=== Numerical Gradient Check ===")
            epsilon = 1e-5
            
            # Check gradient for first element
            i, j = 0, 0
            predictions_plus = predictions.copy()
            predictions_plus[i, j] += epsilon
            loss_plus, _ = mse_loss_forward(predictions_plus, targets)
            
            predictions_minus = predictions.copy()
            predictions_minus[i, j] -= epsilon
            loss_minus, _ = mse_loss_forward(predictions_minus, targets)
            
            numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
            analytical_grad = dpredictions[i, j]
            
            print(f"Element [{i},{j}]:")
            print(f"Numerical gradient: {numerical_grad:.6f}")
            print(f"Analytical gradient: {analytical_grad:.6f}")
            
            error = abs(numerical_grad - analytical_grad)
            print(f"Error: {error:.2e} (should be < 1e-7)")

def test_mse_properties():
    """Test mathematical properties of MSE loss."""
    print("\n=== Testing MSE Properties ===")
    
    # Property 1: Zero loss when predictions equal targets
    predictions = np.array([1.0, 2.0, 3.0])
    targets = predictions.copy()
    loss, _ = mse_loss_forward(predictions, targets)
    
    if loss is not None:
        print(f"1. Perfect predictions: Loss = {loss:.6f} (should be 0)")
        
        # Property 2: Loss is always non-negative
        predictions = np.random.randn(10)
        targets = np.random.randn(10)
        loss, _ = mse_loss_forward(predictions, targets)
        print(f"2. Random data: Loss = {loss:.4f} (should be >= 0)")
        
        # Property 3: Scaling error scales loss quadratically
        error = np.array([1.0, 1.0, 1.0])
        loss1, _ = mse_loss_forward(error, np.zeros(3))
        loss2, _ = mse_loss_forward(2*error, np.zeros(3))
        print(f"3. Error scaling: Loss(x) = {loss1:.4f}, Loss(2x) = {loss2:.4f}")
        print(f"   Ratio: {loss2/loss1:.1f} (should be 4.0 for quadratic)")

if __name__ == "__main__":
    test_mse_simple()
    test_mse_batched()
    test_mse_properties()
    
    print("\n=== Key Concepts ===")
    print("1. MSE penalizes large errors more than small ones (quadratic)")
    print("2. Gradient points from prediction toward target")
    print("3. Gradient magnitude is proportional to error")
    print("4. Common for regression tasks (predicting continuous values)")