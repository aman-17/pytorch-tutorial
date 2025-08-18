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
        # Initialize multi-head attention parameters
        # W_q, W_k, W_v: (d_model, d_model) each
        # W_o: (d_model, d_model) - output projection
        std_attn = np.sqrt(2.0 / (self.d_model + self.d_model))
        self.W_q = np.random.randn(self.d_model, self.d_model) * std_attn
        self.W_k = np.random.randn(self.d_model, self.d_model) * std_attn
        self.W_v = np.random.randn(self.d_model, self.d_model) * std_attn
        self.W_o = np.random.randn(self.d_model, self.d_model) * std_attn
        
        # Initialize feed-forward parameters  
        # W1: (d_model, d_ff) - first linear layer
        # b1: (d_ff,) - first bias
        # W2: (d_ff, d_model) - second linear layer  
        # b2: (d_model,) - second bias
        std_ff1 = np.sqrt(2.0 / (self.d_model + self.d_ff))
        std_ff2 = np.sqrt(2.0 / (self.d_ff + self.d_model))
        self.W1 = np.random.randn(self.d_model, self.d_ff) * std_ff1
        self.b1 = np.zeros(self.d_ff)
        self.W2 = np.random.randn(self.d_ff, self.d_model) * std_ff2
        self.b2 = np.zeros(self.d_model)
        
        # Initialize layer normalization parameters
        # ln1_gamma, ln1_beta: (d_model,) - first layer norm
        # ln2_gamma, ln2_beta: (d_model,) - second layer norm
        self.ln1_gamma = np.ones(self.d_model)
        self.ln1_beta = np.zeros(self.d_model)
        self.ln2_gamma = np.ones(self.d_model)
        self.ln2_beta = np.zeros(self.d_model)
    
    def layer_norm_forward(self, x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, 
                          eps: float = 1e-5) -> Tuple[np.ndarray, Dict]:
        """
        Forward pass for layer normalization.
        x: (batch_size, seq_len, d_model)
        Returns: normalized output and cache for backward pass
        """
        # Implement layer normalization forward pass
        # 1. Compute mean and variance along last dimension
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        # 2. Normalize: (x - mean) / sqrt(var + eps)
        normalized = (x - mean) / np.sqrt(var + eps)
        
        # 3. Scale and shift: gamma * normalized + beta
        output = gamma * normalized + beta
        
        # 4. Cache intermediate values for backward pass
        cache = {
            'x': x,
            'gamma': gamma,
            'beta': beta,
            'mean': mean,
            'var': var,
            'normalized': normalized,
            'eps': eps
        }
        
        return output, cache
    
    def layer_norm_backward(self, dout: np.ndarray, cache: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass for layer normalization.
        Returns: dx, dgamma, dbeta
        """
        # Implement layer normalization backward pass
        x, gamma, mean, var, normalized, eps = cache['x'], cache['gamma'], cache['mean'], cache['var'], cache['normalized'], cache['eps']
        
        # Key gradients to compute:
        # dgamma = sum(dout * normalized, axis=(0,1))
        dgamma = np.sum(dout * normalized, axis=(0, 1))
        # dbeta = sum(dout, axis=(0,1))
        dbeta = np.sum(dout, axis=(0, 1))
        
        # dx requires computing gradients through normalization operation
        N = x.shape[0] * x.shape[1]  # batch_size * seq_len
        d_normalized = dout * gamma
        
        # Gradient through normalization
        inv_std = 1.0 / np.sqrt(var + eps)
        dx_centered = d_normalized * inv_std
        
        d_var = np.sum(d_normalized * (x - mean), axis=(0, 1), keepdims=True) * (-0.5) * (inv_std ** 3)
        d_mean = np.sum(dx_centered, axis=(0, 1), keepdims=True) * (-1) + d_var * np.sum(-2.0 * (x - mean), axis=(0, 1), keepdims=True) / N
        
        dx = dx_centered + d_var * 2.0 * (x - mean) / N + d_mean / N
        
        return dx, dgamma, dbeta
    
    def attention_forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Multi-head attention forward pass.
        x: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Compute Q, K, V projections
        Q = np.matmul(x, self.W_q)
        K = np.matmul(x, self.W_k)
        V = np.matmul(x, self.W_v)
        
        # Reshape for multi-head attention
        # Shape: (batch_size, seq_len, n_heads, d_k)
        # Then transpose to: (batch_size, n_heads, seq_len, d_k)
        Q_heads = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K_heads = K.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V_heads = V.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Compute attention scores
        # scores = Q @ K^T / sqrt(d_k)
        scores = np.matmul(Q_heads, np.swapaxes(K_heads, -2, -1)) / np.sqrt(self.d_k)
        
        # Apply softmax to get attention weights
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # Apply attention to values
        attn_output = np.matmul(attn_weights, V_heads)
        
        # Reshape and apply output projection
        # Concatenate heads and project: output @ W_o
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        output = np.matmul(attn_output, self.W_o)
        
        cache = {
            'x': x,
            'Q': Q,
            'K': K,
            'V': V,
            'Q_heads': Q_heads,
            'K_heads': K_heads,
            'V_heads': V_heads,
            'scores': scores,
            'attn_weights': attn_weights,
            'attn_output': attn_output
        }
        
        return output, cache
    
    def attention_backward(self, dout: np.ndarray, cache: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Multi-head attention backward pass.
        Returns: dx and parameter gradients
        """
        # Implement attention backward pass
        x, Q, K, V = cache['x'], cache['Q'], cache['K'], cache['V']
        Q_heads, K_heads, V_heads = cache['Q_heads'], cache['K_heads'], cache['V_heads']
        scores, attn_weights, attn_output = cache['scores'], cache['attn_weights'], cache['attn_output']
        
        # 1. Output projection (W_o)
        # attn_output: (batch_size, seq_len, d_model), dout: (batch_size, seq_len, d_model)
        dW_o = np.matmul(attn_output.swapaxes(-2, -1), dout)  # (batch_size, d_model, d_model)
        dW_o = np.mean(dW_o, axis=0)  # Average over batch: (d_model, d_model)
        d_attn_output = np.matmul(dout, self.W_o.T)
        
        # 2. Concatenation and reshaping
        batch_size, seq_len, d_model = d_attn_output.shape
        d_attn_heads = d_attn_output.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # 3. Attention application (attn_weights @ V)
        dV_heads = np.matmul(np.swapaxes(attn_weights, -2, -1), d_attn_heads)
        d_attn_weights = np.matmul(d_attn_heads, np.swapaxes(V_heads, -2, -1))
        
        # 4. Softmax operation (tricky!)
        # dsoftmax = softmax * (dout - sum(dout * softmax, axis=-1, keepdims=True))
        softmax_sum = np.sum(d_attn_weights * attn_weights, axis=-1, keepdims=True)
        d_scores = attn_weights * (d_attn_weights - softmax_sum)
        
        # 5. Scaled dot-product (Q @ K^T / sqrt(d_k))
        dQ_heads = np.matmul(d_scores, K_heads) / np.sqrt(self.d_k)
        dK_heads = np.matmul(np.swapaxes(d_scores, -2, -1), Q_heads) / np.sqrt(self.d_k)
        
        # 6. Q, K, V projections - reshape back
        dQ = dQ_heads.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        dK = dK_heads.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        dV = dV_heads.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
        
        # Parameter gradients - average over batch
        dW_q = np.matmul(x.swapaxes(-2, -1), dQ)  # (batch_size, d_model, d_model)
        dW_q = np.mean(dW_q, axis=0)  # (d_model, d_model)
        dW_k = np.matmul(x.swapaxes(-2, -1), dK)
        dW_k = np.mean(dW_k, axis=0)
        dW_v = np.matmul(x.swapaxes(-2, -1), dV)
        dW_v = np.mean(dW_v, axis=0)
        
        # Input gradients
        dx_q = np.matmul(dQ, self.W_q.T)
        dx_k = np.matmul(dK, self.W_k.T)
        dx_v = np.matmul(dV, self.W_v.T)
        dx = dx_q + dx_k + dx_v
        
        grads = {
            'dW_q': dW_q,
            'dW_k': dW_k,
            'dW_v': dW_v,
            'dW_o': dW_o
        }
        
        return dx, grads
    
    def feedforward_forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Feed-forward network forward pass.
        x: (batch_size, seq_len, d_model)
        """
        # Implement feed-forward forward pass
        # 1. First linear: x @ W1 + b1
        hidden_pre_relu = np.matmul(x, self.W1) + self.b1
        
        # 2. ReLU activation
        hidden_post_relu = np.maximum(0, hidden_pre_relu)
        
        # 3. Second linear: hidden @ W2 + b2
        output = np.matmul(hidden_post_relu, self.W2) + self.b2
        
        cache = {
            'x': x,
            'hidden_pre_relu': hidden_pre_relu,
            'hidden_post_relu': hidden_post_relu
        }
        
        return output, cache
    
    def feedforward_backward(self, dout: np.ndarray, cache: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Feed-forward network backward pass.
        """
        # Implement feed-forward backward pass
        x, hidden_pre_relu, hidden_post_relu = cache['x'], cache['hidden_pre_relu'], cache['hidden_post_relu']
        
        # 1. Gradient through second linear layer
        dW2 = np.matmul(hidden_post_relu.swapaxes(-2, -1), dout)  # (batch_size, d_ff, d_model)
        dW2 = np.mean(dW2, axis=0)  # Average over batch: (d_ff, d_model)
        db2 = np.sum(dout, axis=(0, 1))
        d_hidden_post_relu = np.matmul(dout, self.W2.T)
        
        # 2. Gradient through ReLU (zero where input was negative)
        d_hidden_pre_relu = d_hidden_post_relu * (hidden_pre_relu > 0).astype(float)
        
        # 3. Gradient through first linear layer
        dW1 = np.matmul(x.swapaxes(-2, -1), d_hidden_pre_relu)  # (batch_size, d_model, d_ff)
        dW1 = np.mean(dW1, axis=0)  # Average over batch: (d_model, d_ff)
        db1 = np.sum(d_hidden_pre_relu, axis=(0, 1))
        dx = np.matmul(d_hidden_pre_relu, self.W1.T)
        
        grads = {
            'dW1': dW1,
            'db1': db1,
            'dW2': dW2,
            'db2': db2
        }
        
        return dx, grads
    
    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Complete transformer layer forward pass.
        x: (batch_size, seq_len, d_model)
        """
        # Implement complete forward pass
        # 1. First residual block: x + attention(layer_norm(x))
        ln1_out, ln1_cache = self.layer_norm_forward(x, self.ln1_gamma, self.ln1_beta)
        attn_out, attn_cache = self.attention_forward(ln1_out)
        attn_residual = x + attn_out
        
        # 2. Second residual block: x + feedforward(layer_norm(x))
        ln2_out, ln2_cache = self.layer_norm_forward(attn_residual, self.ln2_gamma, self.ln2_beta)
        ff_out, ff_cache = self.feedforward_forward(ln2_out)
        output = attn_residual + ff_out
        
        # Store all intermediate values and caches for backward pass
        forward_cache = {
            'input': x,
            'ln1_cache': ln1_cache,
            'attn_cache': attn_cache,
            'ln2_cache': ln2_cache,
            'ff_cache': ff_cache,
            'attn_residual': attn_residual,
            'ln1_out': ln1_out,
            'ln2_out': ln2_out,
            'attn_out': attn_out,
            'ff_out': ff_out
        }
        
        return output, forward_cache
    
    def backward(self, dout: np.ndarray, forward_cache: Dict) -> Tuple[np.ndarray, Dict]:
        """
        Complete transformer layer backward pass.
        Returns input gradient and all parameter gradients.
        """
        # Implement complete backward pass
        # Work backwards through the computation graph:
        
        # 1. Gradient through second residual connection
        d_attn_residual = dout.copy()
        d_ff_out = dout.copy()
        
        # 2. Gradient through feed-forward block
        d_ln2_out, ff_grads = self.feedforward_backward(d_ff_out, forward_cache['ff_cache'])
        
        # 3. Gradient through second layer norm
        d_attn_residual_2, dln2_gamma, dln2_beta = self.layer_norm_backward(d_ln2_out, forward_cache['ln2_cache'])
        d_attn_residual += d_attn_residual_2
        
        # 4. Gradient through first residual connection  
        dx_1 = d_attn_residual.copy()
        d_attn_out = d_attn_residual.copy()
        
        # 5. Gradient through attention block
        d_ln1_out, attn_grads = self.attention_backward(d_attn_out, forward_cache['attn_cache'])
        
        # 6. Gradient through first layer norm
        dx_2, dln1_gamma, dln1_beta = self.layer_norm_backward(d_ln1_out, forward_cache['ln1_cache'])
        
        # Combine input gradients
        dx = dx_1 + dx_2
        
        all_grads = {
            # Attention gradients
            'dW_q': attn_grads['dW_q'],
            'dW_k': attn_grads['dW_k'], 
            'dW_v': attn_grads['dW_v'],
            'dW_o': attn_grads['dW_o'],
            
            # Feed-forward gradients
            'dW1': ff_grads['dW1'],
            'db1': ff_grads['db1'],
            'dW2': ff_grads['dW2'],
            'db2': ff_grads['db2'],
            
            # Layer norm gradients
            'dln1_gamma': dln1_gamma,
            'dln1_beta': dln1_beta,
            'dln2_gamma': dln2_gamma,
            'dln2_beta': dln2_beta
        }
        
        return dx, all_grads

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
    
    # Simple gradient check for a few parameters
    print("Testing a few parameters...")
    
    # Test W1 parameter
    if grads_analytical['dW1'] is not None:
        i, j = 0, 0  # Test first element
        
        # Store original value
        original_val = layer.W1[i, j]
        
        # Compute loss with +epsilon
        layer.W1[i, j] = original_val + epsilon
        out_plus, _ = layer.forward(x)
        loss_plus = np.sum(out_plus)
        
        # Compute loss with -epsilon
        layer.W1[i, j] = original_val - epsilon
        out_minus, _ = layer.forward(x)
        loss_minus = np.sum(out_minus)
        
        # Restore original value
        layer.W1[i, j] = original_val
        
        # Numerical gradient
        numerical_grad = (loss_plus - loss_minus) / (2 * epsilon)
        analytical = grads_analytical['dW1'][i, j]
        
        relative_error = abs(numerical_grad - analytical) / (abs(numerical_grad) + abs(analytical) + 1e-8)
        
        status = "✓" if relative_error < 1e-3 else "✗"
        print(f"{status} W1[{i},{j}]: analytical={analytical:.6f}, numerical={numerical_grad:.6f}, error={relative_error:.2e}")
    
    print("Gradient check completed (simplified version)!")

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