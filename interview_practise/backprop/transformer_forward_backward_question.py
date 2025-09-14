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

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional
import math

class ManualTransformerLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
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
        std_attn = math.sqrt(2.0 / (self.d_model + self.d_model))
        self.W_q = nn.Parameter(torch.randn(self.d_model, self.d_model) * std_attn)
        self.W_k = nn.Parameter(torch.randn(self.d_model, self.d_model) * std_attn)
        self.W_v = nn.Parameter(torch.randn(self.d_model, self.d_model) * std_attn)
        self.W_o = nn.Parameter(torch.randn(self.d_model, self.d_model) * std_attn)
        
        # Initialize feed-forward parameters  
        # W1: (d_model, d_ff) - first linear layer
        # b1: (d_ff,) - first bias
        # W2: (d_ff, d_model) - second linear layer  
        # b2: (d_model,) - second bias
        std_ff1 = math.sqrt(2.0 / (self.d_model + self.d_ff))
        std_ff2 = math.sqrt(2.0 / (self.d_ff + self.d_model))
        self.W1 = nn.Parameter(torch.randn(self.d_model, self.d_ff) * std_ff1)
        self.b1 = nn.Parameter(torch.zeros(self.d_ff))
        self.W2 = nn.Parameter(torch.randn(self.d_ff, self.d_model) * std_ff2)
        self.b2 = nn.Parameter(torch.zeros(self.d_model))
        
        # Initialize layer normalization parameters
        # ln1_gamma, ln1_beta: (d_model,) - first layer norm
        # ln2_gamma, ln2_beta: (d_model,) - second layer norm
        self.ln1_gamma = nn.Parameter(torch.ones(self.d_model))
        self.ln1_beta = nn.Parameter(torch.zeros(self.d_model))
        self.ln2_gamma = nn.Parameter(torch.ones(self.d_model))
        self.ln2_beta = nn.Parameter(torch.zeros(self.d_model))
    
    def layer_norm_forward(self, x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor, 
                          eps: float = 1e-5) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass for layer normalization.
        x: (batch_size, seq_len, d_model)
        Returns: normalized output and cache for backward pass
        """
        # Implement layer normalization forward pass
        # 1. Compute mean and variance along last dimension
        mean = torch.mean(x, dim=-1, keepdim=True)
        var = torch.var(x, dim=-1, keepdim=True, unbiased=False)
        
        # 2. Normalize: (x - mean) / sqrt(var + eps)
        normalized = (x - mean) / torch.sqrt(var + eps)
        
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
    
    def layer_norm_backward(self, dout: torch.Tensor, cache: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Backward pass for layer normalization.
        Returns: dx, dgamma, dbeta
        """
        # Implement layer normalization backward pass
        x, gamma, mean, var, normalized, eps = cache['x'], cache['gamma'], cache['mean'], cache['var'], cache['normalized'], cache['eps']
        
        # Key gradients to compute:
        # dgamma = sum(dout * normalized, dim=(0,1))
        dgamma = torch.sum(dout * normalized, dim=(0, 1))
        # dbeta = sum(dout, dim=(0,1))
        dbeta = torch.sum(dout, dim=(0, 1))
        
        # dx requires computing gradients through normalization operation
        N = x.shape[0] * x.shape[1]  # batch_size * seq_len
        d_normalized = dout * gamma
        
        # Gradient through normalization
        inv_std = 1.0 / torch.sqrt(var + eps)
        dx_centered = d_normalized * inv_std
        
        d_var = torch.sum(d_normalized * (x - mean), dim=(0, 1), keepdim=True) * (-0.5) * (inv_std ** 3)
        d_mean = torch.sum(dx_centered, dim=(0, 1), keepdim=True) * (-1) + d_var * torch.sum(-2.0 * (x - mean), dim=(0, 1), keepdim=True) / N
        
        dx = dx_centered + d_var * 2.0 * (x - mean) / N + d_mean / N
        
        return dx, dgamma, dbeta
    
    def attention_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Multi-head attention forward pass.
        x: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Compute Q, K, V projections
        Q = torch.matmul(x, self.W_q)
        K = torch.matmul(x, self.W_k)
        V = torch.matmul(x, self.W_v)
        
        # Reshape for multi-head attention
        # Shape: (batch_size, seq_len, n_heads, d_k)
        # Then transpose to: (batch_size, n_heads, seq_len, d_k)
        Q_heads = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K_heads = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V_heads = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Compute attention scores
        # scores = Q @ K^T / sqrt(d_k)
        scores = torch.matmul(Q_heads, K_heads.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V_heads)
        
        # Reshape and apply output projection
        # Concatenate heads and project: output @ W_o
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = torch.matmul(attn_output, self.W_o)
        
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
    
    def attention_backward(self, dout: torch.Tensor, cache: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Multi-head attention backward pass.
        Returns: dx and parameter gradients
        """
        # Implement attention backward pass
        x, Q, K, V = cache['x'], cache['Q'], cache['K'], cache['V']
        Q_heads, K_heads, V_heads = cache['Q_heads'], cache['K_heads'], cache['V_heads']
        scores, attn_weights, attn_output = cache['scores'], cache['attn_weights'], cache['attn_output']
        
        # 1. Output projection (W_o)
        dW_o = torch.matmul(attn_output.transpose(-2, -1), dout)
        d_attn_output = torch.matmul(dout, self.W_o.transpose(-2, -1))
        
        # 2. Concatenation and reshaping
        batch_size, seq_len, d_model = d_attn_output.shape
        d_attn_heads = d_attn_output.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # 3. Attention application (attn_weights @ V)
        dV_heads = torch.matmul(attn_weights.transpose(-2, -1), d_attn_heads)
        d_attn_weights = torch.matmul(d_attn_heads, V_heads.transpose(-2, -1))
        
        # 4. Softmax operation
        # dsoftmax = softmax * (dout - (dout * softmax).sum(dim=-1, keepdim=True))
        softmax_sum = torch.sum(d_attn_weights * attn_weights, dim=-1, keepdim=True)
        d_scores = attn_weights * (d_attn_weights - softmax_sum)
        
        # 5. Scaled dot-product (Q @ K^T / sqrt(d_k))
        dQ_heads = torch.matmul(d_scores, K_heads) / math.sqrt(self.d_k)
        dK_heads = torch.matmul(d_scores.transpose(-2, -1), Q_heads) / math.sqrt(self.d_k)
        
        # 6. Q, K, V projections - reshape back
        dQ = dQ_heads.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        dK = dK_heads.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        dV = dV_heads.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Parameter gradients
        dW_q = torch.matmul(x.transpose(-2, -1), dQ)
        dW_k = torch.matmul(x.transpose(-2, -1), dK)
        dW_v = torch.matmul(x.transpose(-2, -1), dV)
        
        # Input gradients
        dx_q = torch.matmul(dQ, self.W_q.transpose(-2, -1))
        dx_k = torch.matmul(dK, self.W_k.transpose(-2, -1))
        dx_v = torch.matmul(dV, self.W_v.transpose(-2, -1))
        dx = dx_q + dx_k + dx_v
        
        grads = {
            'dW_q': dW_q,
            'dW_k': dW_k,
            'dW_v': dW_v,
            'dW_o': dW_o
        }
        
        return dx, grads
    
    def feedforward_forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """
        Feed-forward network forward pass.
        x: (batch_size, seq_len, d_model)
        """
        # Implement feed-forward forward pass
        # 1. First linear: x @ W1 + b1
        hidden_pre_relu = torch.matmul(x, self.W1) + self.b1
        
        # 2. ReLU activation: F.relu
        hidden_post_relu = F.relu(hidden_pre_relu)
        
        # 3. Second linear: hidden @ W2 + b2
        output = torch.matmul(hidden_post_relu, self.W2) + self.b2
        
        cache = {
            'x': x,
            'hidden_pre_relu': hidden_pre_relu,
            'hidden_post_relu': hidden_post_relu
        }
        
        return output, cache
    
    def feedforward_backward(self, dout: torch.Tensor, cache: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Feed-forward network backward pass.
        """
        # Implement feed-forward backward pass
        x, hidden_pre_relu, hidden_post_relu = cache['x'], cache['hidden_pre_relu'], cache['hidden_post_relu']
        
        # 1. Gradient through second linear layer
        dW2 = torch.matmul(hidden_post_relu.transpose(-2, -1), dout)
        db2 = torch.sum(dout, dim=(0, 1))
        d_hidden_post_relu = torch.matmul(dout, self.W2.transpose(-2, -1))
        
        # 2. Gradient through ReLU (zero where input was negative)
        d_hidden_pre_relu = d_hidden_post_relu * (hidden_pre_relu > 0).float()
        
        # 3. Gradient through first linear layer
        dW1 = torch.matmul(x.transpose(-2, -1), d_hidden_pre_relu)
        db1 = torch.sum(d_hidden_pre_relu, dim=(0, 1))
        dx = torch.matmul(d_hidden_pre_relu, self.W1.transpose(-2, -1))
        
        grads = {
            'dW1': dW1,
            'db1': db1,
            'dW2': dW2,
            'db2': db2
        }
        
        return dx, grads
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
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
    
    def backward(self, dout: torch.Tensor, forward_cache: Dict) -> Tuple[torch.Tensor, Dict]:
        """
        Complete transformer layer backward pass.
        Returns input gradient and all parameter gradients.
        """
        # Implement complete backward pass
        # Work backwards through the computation graph:
        
        # 1. Gradient through second residual connection
        d_attn_residual = dout.clone()
        d_ff_out = dout.clone()
        
        # 2. Gradient through feed-forward block
        d_ln2_out, ff_grads = self.feedforward_backward(d_ff_out, forward_cache['ff_cache'])
        
        # 3. Gradient through second layer norm
        d_attn_residual_2, dln2_gamma, dln2_beta = self.layer_norm_backward(d_ln2_out, forward_cache['ln2_cache'])
        d_attn_residual += d_attn_residual_2
        
        # 4. Gradient through first residual connection  
        dx_1 = d_attn_residual.clone()
        d_attn_out = d_attn_residual.clone()
        
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

def numerical_gradient_check(layer: ManualTransformerLayer, x: torch.Tensor, epsilon: float = 1e-5):
    """
    Numerical gradient checking to verify backward pass implementation.
    """
    print("=== Numerical Gradient Check ===")
    
    # Forward pass
    output, cache = layer.forward(x)
    if output is None:
        print("Forward pass not implemented yet")
        return
    
    try:
        # Use torch.autograd.gradcheck for automatic gradient verification
        def transformer_loss(input_tensor):
            out, _ = layer.forward(input_tensor)
            return torch.sum(out ** 2)
        
        # Test gradient correctness
        test_passed = torch.autograd.gradcheck(transformer_loss, x, eps=epsilon, atol=1e-3)
        
        if test_passed:
            print("✓ Gradient check passed!")
        else:
            print("✗ Gradient check failed!")
            
    except Exception as e:
        print(f"✗ Gradient check failed with error: {e}")
        
        # Manual comparison with torch.autograd
        print("\nPerforming manual gradient comparison...")
        
        # Create dummy loss (sum of outputs)
        loss = torch.sum(output ** 2)
        dout = 2 * output
        
        # Analytical gradients
        dx_analytical, grads_analytical = layer.backward(dout, cache)
        
        # Compare with torch.autograd computed gradients
        loss.backward(retain_graph=True)
        
        if x.grad is not None:
            rel_error = torch.norm(dx_analytical - x.grad) / (torch.norm(dx_analytical) + torch.norm(x.grad) + 1e-8)
            print(f"Input gradient relative error: {rel_error.item():.2e}")
        
        # Compare parameter gradients
        for name, param in layer.named_parameters():
            if param.grad is not None and name.replace('.', '_') in grads_analytical:
                grad_name = 'd' + name.replace('.', '_')
                if grad_name in grads_analytical:
                    analytical_grad = grads_analytical[grad_name]
                    if analytical_grad is not None:
                        rel_error = torch.norm(analytical_grad - param.grad) / (torch.norm(analytical_grad) + torch.norm(param.grad) + 1e-8)
                        print(f"{name} gradient relative error: {rel_error.item():.2e}")

def test_individual_components():
    """Test each component separately for easier debugging."""
    d_model, n_heads, d_ff = 64, 4, 256
    batch_size, seq_len = 2, 8
    
    layer = ManualTransformerLayer(d_model, n_heads, d_ff)
    x = torch.randn(batch_size, seq_len, d_model)
    
    print("=== Testing Individual Components ===")
    
    # Test layer normalization
    print("1. Testing Layer Normalization...")
    gamma = torch.ones(d_model)
    beta = torch.zeros(d_model)
    ln_out, ln_cache = layer.layer_norm_forward(x, gamma, beta)
    
    if ln_out is not None:
        print(f"   ✓ Layer norm output shape: {ln_out.shape}")
        print(f"   Output mean: {torch.mean(ln_out).item():.6f} (should be ~0)")
        print(f"   Output std: {torch.std(ln_out).item():.6f} (should be ~1)")
    else:
        print("   ✗ Layer normalization not implemented yet")
    
    # Test attention
    print("2. Testing Multi-Head Attention...")
    attn_out, attn_cache = layer.attention_forward(x)
    if attn_out is not None:
        print(f"   ✓ Attention output shape: {attn_out.shape}")
    else:
        print("   ✗ Attention not implemented yet")
    
    # Test feed-forward
    print("3. Testing Feed-Forward Network...")
    ff_out, ff_cache = layer.feedforward_forward(x)
    if ff_out is not None:
        print(f"   ✓ Feed-forward output shape: {ff_out.shape}")
    else:
        print("   ✗ Feed-forward not implemented yet")

def compare_with_pytorch_transformer():
    """Compare implementation with PyTorch's built-in TransformerEncoderLayer."""
    d_model, n_heads, d_ff = 64, 4, 256
    batch_size, seq_len = 2, 8
    
    print("=== Comparing with PyTorch TransformerEncoderLayer ===")
    
    # Manual implementation
    manual_layer = ManualTransformerLayer(d_model, n_heads, d_ff)
    
    # PyTorch implementation
    pytorch_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout=0.0, batch_first=True)
    
    # Test input
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    
    print("Manual layer parameters:")
    total_manual_params = 0
    for name, param in manual_layer.named_parameters():
        print(f"   {name}: {param.shape}")
        total_manual_params += param.numel()
    print(f"   Total parameters: {total_manual_params}")
    
    print("\nPyTorch layer parameters:")
    total_pytorch_params = 0
    for name, param in pytorch_layer.named_parameters():
        print(f"   {name}: {param.shape}")
        total_pytorch_params += param.numel()
    print(f"   Total parameters: {total_pytorch_params}")
    
    # Forward pass comparison
    print("\n=== Forward Pass Comparison ===")
    manual_output, _ = manual_layer.forward(x)
    pytorch_output = pytorch_layer(x)
    
    if manual_output is not None:
        print(f"Manual output shape: {manual_output.shape}")
        print(f"PyTorch output shape: {pytorch_output.shape}")
        print(f"Manual output range: [{torch.min(manual_output).item():.3f}, {torch.max(manual_output).item():.3f}]")
        print(f"PyTorch output range: [{torch.min(pytorch_output).item():.3f}, {torch.max(pytorch_output).item():.3f}]")
    else:
        print("Manual implementation not working yet")

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
    x = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
    
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
        print(f"Output range: [{torch.min(output).item():.3f}, {torch.max(output).item():.3f}]")
        
        # Test backward pass
        print("\n=== Testing Complete Backward Pass ===")
        dout = torch.randn(*output.shape)
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
    
    # Compare with PyTorch implementation
    print("\n=== Comparing with PyTorch ===")
    compare_with_pytorch_transformer()
    
    print("\n=== Implementation Notes ===")
    print("Key challenges in this implementation:")
    print("1. Softmax gradient in attention mechanism")
    print("2. Layer normalization backward pass")
    print("3. Proper gradient accumulation through residual connections")
    print("4. Multi-head attention reshaping and gradient flow")
    print("5. Numerical stability in all operations")
    print("6. Use torch.autograd.gradcheck for gradient verification")
    print("7. Compare with nn.TransformerEncoderLayer for validation")