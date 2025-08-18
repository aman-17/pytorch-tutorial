"""
Question: Transformer Forward and Backward Propagation from Scratch (Hard)

Implement a complete transformer layer with manual forward and backward propagation.
You'll implement the mathematical operations behind attention, feed-forward networks,
and layer normalization, along with their gradients for backpropagation.

This question tests deep understanding of:
1. Multi-head attention mathematics and gradients
2. Layer normalization forward/backward pass
3. Feed-forward network gradients
4. Residual connections and gradient flow
5. Manual computation graphs and chain rule application

Your task: Implement all components with both forward and backward passes.
"""

import numpy as np
from typing import Tuple, Dict, Optional

class ManualTransformerLayer:
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_ff = d_ff
        self.dropout = dropout
        
        # Initialize parameters
        self._init_parameters()
        
        # Cache for backward pass
        self.cache = {}
        
    def _init_parameters(self):
        """Initialize all transformer parameters using Xavier/Glorot initialization"""
        # TODO: Initialize multi-head attention parameters
        # W_q, W_k, W_v: (d_model, d_model) each
        # W_o: (d_model, d_model) - output projection
        
        # TODO: Initialize feed-forward parameters  
        # W1: (d_model, d_ff) - first linear layer
        # b1: (d_ff,) - first bias
        # W2: (d_ff, d_model) - second linear layer  
        # b2: (d_model,) - second bias
        
        # TODO: Initialize layer normalization parameters
        # ln1_gamma, ln1_beta: (d_model,) - first layer norm
        # ln2_gamma, ln2_beta: (d_model,) - second layer norm
        
        # Use Xavier initialization: std = sqrt(2 / (fan_in + fan_out))
        pass
    
    def layer_norm_forward(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, 
                          eps: float = 1e-5) -> Tuple[np.ndarray, Dict]:
        """
        Forward pass for layer normalization.
        x: (batch_size, seq_len, d_model)
        Returns: normalized output and cache for backward pass
        """
        # TODO: Implement layer normalization forward pass
        # 1. Compute mean and variance along last dimension
        # 2. Normalize: (x - mean) / sqrt(var + eps)
        # 3. Scale and shift: gamma * normalized + beta
        # 4. Cache intermediate values for backward pass
        
        cache = {
            'x': x,
            'gamma': gamma,
            'beta': beta,
            'mean': None,  # TODO: compute
            'var': None,   # TODO: compute
            'normalized': None,  # TODO: compute
            'eps': eps
        }
        
        return None, cache  # TODO: return (output, cache)
    
    def layer_norm_backward(self, dout: np.ndarray, cache: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass for layer normalization.
        Returns: dx, dgamma, dbeta
        """
        # TODO: Implement layer normalization backward pass
        # This involves careful application of chain rule for normalization
        # 
        # Key gradients to compute:
        # dgamma = sum(dout * normalized, axis=(0,1))
        # dbeta = sum(dout, axis=(0,1))
        # dx requires computing gradients through normalization operation
        
        return None, None, None  # TODO: return (dx, dgamma, dbeta)
    
    def attention_forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Multi-head attention forward pass.
        x: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # TODO: Compute Q, K, V projections
        #  Q = x @ W_q, K = x @ W_k, V = x @ W_v
        
        # TODO: Reshape for multi-head attention
        # Shape: (batch_size, seq_len, n_heads, d_k)
        # Then transpose to: (batch_size, n_heads, seq_len, d_k)
        
        # TODO: Compute attention scores
        # scores = Q @ K^T / sqrt(d_k)
        
        # TODO: Apply softmax to get attention weights
        # attn_weights = softmax(scores, axis=-1)
        
        # TODO: Apply attention to values
        # attn_output = attn_weights @ V
        
        # TODO: Reshape and apply output projection
        # Concatenate heads and project: output @ W_o
        
        cache = {
            'x': x,
            'Q': None,  # TODO: store computed values
            'K': None,
            'V': None,
            'scores': None,
            'attn_weights': None,
            'attn_output': None
        }
        
        return None, cache  # TODO: return (output, cache)
    
    def attention_backward(self, dout: np.ndarray, cache: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Multi-head attention backward pass.
        Returns: dx and parameter gradients
        """
        # TODO: Implement attention backward pass
        # This is the most complex part - gradients flow through:
        # 1. Output projection (W_o)
        # 2. Concatenation and reshaping
        # 3. Attention application (attn_weights @ V)
        # 4. Softmax operation (tricky!)
        # 5. Scaled dot-product (Q @ K^T / sqrt(d_k))
        # 6. Q, K, V projections
        
        # Key insight: Softmax gradient is:
        # dsoftmax = softmax * (dout - sum(dout * softmax, axis=-1, keepdims=True))
        
        grads = {
            'dW_q': None,  # TODO: compute gradients
            'dW_k': None,
            'dW_v': None,
            'dW_o': None
        }
        
        return None, grads  # TODO: return (dx, grads)
    
    def feedforward_forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Feed-forward network forward pass.
        x: (batch_size, seq_len, d_model)
        """
        # TODO: Implement feed-forward forward pass
        # 1. First linear: x @ W1 + b1
        # 2. ReLU activation
        # 3. Second linear: hidden @ W2 + b2
        
        cache = {
            'x': x,
            'hidden_pre_relu': None,  # TODO: store values
            'hidden_post_relu': None
        }
        
        return None, cache  # TODO: return (output, cache)
    
    def feedforward_backward(self, dout: np.ndarray, cache: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Feed-forward network backward pass.
        """
        # TODO: Implement feed-forward backward pass
        # 1. Gradient through second linear layer
        # 2. Gradient through ReLU (zero where input was negative)
        # 3. Gradient through first linear layer
        
        grads = {
            'dW1': None,  # TODO: compute gradients
            'db1': None,
            'dW2': None,
            'db2': None
        }
        
        return None, grads  # TODO: return (dx, grads)
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Complete transformer layer forward pass.
        x: (batch_size, seq_len, d_model)
        """
        # TODO: Implement complete forward pass
        # 1. First residual block: x + attention(layer_norm(x))
        # 2. Second residual block: x + feedforward(layer_norm(x))
        
        # Store all intermediate values and caches for backward pass
        forward_cache = {
            'input': x,
            'ln1_cache': None,
            'attn_cache': None,
            'ln2_cache': None,
            'ff_cache': None,
            'attn_residual': None,  # x + attention output
        }
        
        return None, forward_cache  # TODO: return (output, forward_cache)
    
    def backward(self, dout: np.ndarray, forward_cache: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Complete transformer layer backward pass.
        Returns input gradient and all parameter gradients.
        """
        # TODO: Implement complete backward pass
        # Work backwards through the computation graph:
        # 1. Gradient through second residual connection
        # 2. Gradient through feed-forward block
        # 3. Gradient through second layer norm
        # 4. Gradient through first residual connection  
        # 5. Gradient through attention block
        # 6. Gradient through first layer norm
        
        all_grads = {
            # Attention gradients
            'dW_q': None,
            'dW_k': None, 
            'dW_v': None,
            'dW_o': None,
            
            # Feed-forward gradients
            'dW1': None,
            'db1': None,
            'dW2': None,
            'db2': None,
            
            # Layer norm gradients
            'dln1_gamma': None,
            'dln1_beta': None,
            'dln2_gamma': None,
            'dln2_beta': None
        }
        
        return None, all_grads  # TODO: return (dx, all_grads)

def numerical_gradient_check(layer: ManualTransformerLayer, x: np.ndarray, epsilon: float = 1e-5):
    """
    Numerical gradient checking to verify backward pass implementation.
    """
    # TODO: Implement numerical gradient checking
    # 1. Compute analytical gradients using backward pass
    # 2. Compute numerical gradients using finite differences
    # 3. Compare and report relative error
    
    print("=== Numerical Gradient Check ===")
    
    # Forward pass
    output, cache = layer.forward(x)
    if output is None:
        print("Forward pass not implemented yet")
        return
    
    # Create dummy loss (sum of outputs)
    loss = np.sum(output)
    dout = np.ones_like(output)
    
    # Analytical gradients
    dx_analytical, grads_analytical = layer.backward(dout, cache)
    
    # TODO: For each parameter, compute numerical gradient
    # numerical_grad = (f(param + eps) - f(param - eps)) / (2 * eps)
    
    # TODO: Compare analytical vs numerical gradients
    # relative_error = |analytical - numerical| / (|analytical| + |numerical| + eps)
    
    pass

def test_individual_components():
    """Test each component separately for easier debugging."""
    d_model, n_heads, d_ff = 64, 4, 256
    batch_size, seq_len = 2, 8
    
    layer = ManualTransformerLayer(d_model, n_heads, d_ff)
    x = np.random.randn(batch_size, seq_len, d_model)
    
    print("=== Testing Individual Components ===")
    
    # Test layer normalization
    print("1. Testing Layer Normalization...")
    gamma = np.ones(d_model)
    beta = np.zeros(d_model)
    ln_out, ln_cache = layer.layer_norm_forward(x, gamma, beta)
    
    if ln_out is not None:
        print(f"   Layer norm output shape: {ln_out.shape}")
        print(f"   Output mean: {np.mean(ln_out):.6f} (should be ~0)")
        print(f"   Output std: {np.std(ln_out):.6f} (should be ~1)")
    else:
        print("   Layer normalization not implemented yet")
    
    # Test attention
    print("2. Testing Multi-Head Attention...")
    attn_out, attn_cache = layer.attention_forward(x)
    if attn_out is not None:
        print(f"   Attention output shape: {attn_out.shape}")
    else:
        print("   Attention not implemented yet")
    
    # Test feed-forward
    print("3. Testing Feed-Forward Network...")
    ff_out, ff_cache = layer.feedforward_forward(x)
    if ff_out is not None:
        print(f"   Feed-forward output shape: {ff_out.shape}")
    else:
        print("   Feed-forward not implemented yet")

# Test your implementation
if __name__ == "__main__":
    print("=== Transformer Forward/Backward Implementation Test ===")
    
    # Model parameters
    d_model = 64
    n_heads = 4  
    d_ff = 256
    batch_size = 2
    seq_len = 8
    
    # Create layer and input
    layer = ManualTransformerLayer(d_model, n_heads, d_ff)
    x = np.random.randn(batch_size, seq_len, d_model)
    
    print(f"Input shape: {x.shape}")
    print(f"Model parameters: d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}")
    
    # Test individual components
    test_individual_components()
    
    # Test complete forward pass
    print("\n=== Testing Complete Forward Pass ===")
    output, cache = layer.forward(x)
    
    if output is not None:
        print(f"Forward pass successful!")
        print(f"Output shape: {output.shape}")
        print(f"Output range: [{np.min(output):.3f}, {np.max(output):.3f}]")
        
        # Test backward pass
        print("\n=== Testing Complete Backward Pass ===")
        dout = np.random.randn(*output.shape)
        dx, grads = layer.backward(dout, cache)
        
        if dx is not None:
            print(f"Backward pass successful!")
            print(f"Input gradient shape: {dx.shape}")
            print(f"Number of parameter gradients: {len([k for k, v in grads.items() if v is not None])}")
            
            # Numerical gradient check
            print("\n=== Running Gradient Check ===")
            numerical_gradient_check(layer, x)
        else:
            print("Backward pass not implemented yet")
    else:
        print("Forward pass not implemented yet")
    
    print("\n=== Implementation Notes ===")
    print("Key challenges in this implementation:")
    print("1. Softmax gradient in attention mechanism")
    print("2. Layer normalization backward pass")
    print("3. Proper gradient accumulation through residual connections")
    print("4. Multi-head attention reshaping and gradient flow")
    print("5. Numerical stability in all operations")